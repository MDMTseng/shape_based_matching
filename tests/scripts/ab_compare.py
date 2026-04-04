#!/usr/bin/env python3
"""
A/B comparison script for shape_based_matching regression tests.

Usage:
    python ab_compare.py <ref_A> <ref_B>
    python ab_compare.py <ref_A> <ref_B> --repeat 5
    python ab_compare.py <ref_A> <ref_B> --target "1 2 3"
    python ab_compare.py <ref_A> <ref_B> --build-dir mybuild

Workflow:
    1. Stash dirty changes if needed
    2. Checkout ref_A, build, run test_regression --json, collect metrics
    3. Checkout ref_B, build, run test_regression --json, collect metrics
    4. Compute deltas, write output/ab_diff.csv (and ab_stats.csv if --repeat)
    5. Restore original branch/commit

Exit code: 0 if no regressions, 1 if any metric regressed.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Paths (relative to repo root, using os.path for Windows compat)
# ---------------------------------------------------------------------------
RESULTS_JSON = os.path.join("output", "regression_results.json")
THRESHOLDS_CSV = os.path.join("tests", "test_thresholds.csv")
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled when not a tty or on dumb terminals)
# ---------------------------------------------------------------------------
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code, text):
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text

def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def bold(t):   return _c("1", t)

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def run(cmd, check=True, capture=True):
    """Run a shell command; return stdout string."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr if capture else ""
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}\n{stderr}")
    return result.stdout.strip() if capture else ""


def git_current_ref():
    """Return current branch name, or detached HEAD hash."""
    branch = run("git symbolic-ref --short HEAD", check=False)
    if branch:
        return branch
    return run("git rev-parse HEAD")


def git_is_dirty():
    return run("git status --porcelain") != ""


def git_stash_if_dirty():
    """Stash uncommitted changes; return True if stash was created."""
    if git_is_dirty():
        run("git stash push -m ab_compare_autostash")
        print("  Stashed uncommitted changes.")
        return True
    return False


def git_checkout(ref):
    run(f"git checkout {ref}")


def git_stash_pop():
    run("git stash pop", check=False)

# ---------------------------------------------------------------------------
# Threshold helpers (read CSV so we know direction for improved/regressed)
# ---------------------------------------------------------------------------
def load_thresholds(csv_path=THRESHOLDS_CSV):
    """Return dict: id -> {type, metric, op, threshold, description}."""
    thresholds = {}
    try:
        with open(csv_path, "r", newline="") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 5)
                if len(parts) < 5:
                    continue
                tid, ttype, metric, op, thresh = parts[0], parts[1], parts[2], parts[3], parts[4]
                desc = parts[5] if len(parts) > 5 else ""
                try:
                    thresh_val = float(thresh)
                except ValueError:
                    continue
                thresholds[tid] = {
                    "type": ttype,
                    "metric": metric,
                    "op": op,
                    "threshold": thresh_val,
                    "description": desc,
                }
    except FileNotFoundError:
        print(f"WARNING: thresholds file not found: {csv_path}")
    return thresholds

# ---------------------------------------------------------------------------
# Build & Run
# ---------------------------------------------------------------------------
def build(build_dir):
    """Build test_regression in Release mode. Raises on failure."""
    cmd = f"cmake --build {build_dir} --config Release --target test_regression"
    print(f"  Building: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  BUILD FAILED:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Build failed for current checkout")
    print("  Build OK.")


def run_test(build_dir, target):
    """Run test_regression with --json, return parsed JSON dict."""
    exe = os.path.join(build_dir, "Release", "test_regression.exe")
    if not os.path.isfile(exe):
        # Fallback: maybe not in Release subdir (Linux-style)
        exe_alt = os.path.join(build_dir, "test_regression")
        if os.path.isfile(exe_alt):
            exe = exe_alt
        else:
            raise RuntimeError(f"test_regression executable not found at {exe}")

    if target == "all":
        cmd = f"{exe} all --json"
    else:
        cmd = f"{exe} {target} --json"

    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # test_regression returns 1 on failures — that's OK, we still parse JSON
    if result.returncode not in (0, 1):
        print(f"  WARNING: test exited with code {result.returncode}")

    # Parse output JSON
    json_path = RESULTS_JSON
    if not os.path.isfile(json_path):
        raise RuntimeError(f"JSON output not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data

# ---------------------------------------------------------------------------
# Collect metrics for one ref (possibly multiple runs)
# ---------------------------------------------------------------------------
def collect_metrics(ref, build_dir, target, repeat):
    """
    Checkout ref, build, run test(s), return list of metric dicts.
    Each dict: {metric_id: {"actual": float, "threshold": float, "op": str, "status": str}}
    """
    print(f"\n{'='*60}")
    print(f"Collecting metrics for: {bold(ref)}")
    print(f"{'='*60}")

    git_checkout(ref)

    build(build_dir)

    all_runs = []
    for i in range(repeat):
        if repeat > 1:
            print(f"\n  --- Run {i+1}/{repeat} ---")
        data = run_test(build_dir, target)
        all_runs.append(data)

    return all_runs

# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------
def classify_change(metric_id, val_a, val_b, thresholds):
    """
    Determine if a change is improved, regressed, or unchanged.

    improved  = moved AWAY from threshold (more margin)
    regressed = moved TOWARD threshold (less margin)
    unchanged = delta < 0.1% of val_A
    """
    delta = val_b - val_a

    # Unchanged if negligible
    if val_a != 0 and abs(delta / val_a) < 0.001:
        return "unchanged"
    if val_a == 0 and abs(delta) < 1e-9:
        return "unchanged"

    # Look up threshold direction
    info = thresholds.get(metric_id)
    if info is None:
        # No threshold info — use heuristic: smaller is better for most metrics
        # We cannot determine direction, mark as "changed"
        return "changed"

    op = info["op"]

    # For < or <=: lower actual is better (more margin) -> negative delta = improved
    # For > or >=: higher actual is better -> positive delta = improved
    # For ==: any change away from threshold is regressed
    if op in ("<", "<="):
        return "improved" if delta < 0 else "regressed"
    elif op in (">", ">="):
        return "improved" if delta > 0 else "regressed"
    elif op == "==":
        # For equality checks: moving away from target is regressed
        dist_a = abs(val_a - info["threshold"])
        dist_b = abs(val_b - info["threshold"])
        if dist_b < dist_a:
            return "improved"
        elif dist_b > dist_a:
            return "regressed"
        return "unchanged"

    return "changed"


def compute_single_diff(data_a, data_b, thresholds):
    """
    Compare two single-run results.
    Returns list of dicts with: metric, val_A, val_B, delta, delta_pct, status.
    """
    metrics_a = data_a.get("metrics", {})
    metrics_b = data_b.get("metrics", {})

    all_ids = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

    rows = []
    for mid in all_ids:
        ma = metrics_a.get(mid)
        mb = metrics_b.get(mid)
        if ma is None or mb is None:
            # Metric only present in one version
            rows.append({
                "metric": mid,
                "val_A": ma["actual"] if ma else "N/A",
                "val_B": mb["actual"] if mb else "N/A",
                "delta": "N/A",
                "delta_pct": "N/A",
                "status": "missing_in_" + ("B" if mb is None else "A"),
            })
            continue

        val_a = ma["actual"]
        val_b = mb["actual"]
        delta = val_b - val_a
        if val_a != 0:
            delta_pct = 100.0 * delta / abs(val_a)
        else:
            delta_pct = 0.0 if delta == 0 else float("inf")

        status = classify_change(mid, val_a, val_b, thresholds)

        rows.append({
            "metric": mid,
            "val_A": val_a,
            "val_B": val_b,
            "delta": delta,
            "delta_pct": delta_pct,
            "status": status,
        })

    return rows


def compute_stats_diff(runs_a, runs_b, thresholds):
    """
    Compare multiple runs with statistical analysis.
    Returns (diff_rows, stats_rows).
    """
    # Aggregate per-metric across runs
    def aggregate(runs):
        agg = {}  # metric_id -> list of actual values
        for data in runs:
            for mid, minfo in data.get("metrics", {}).items():
                agg.setdefault(mid, []).append(minfo["actual"])
        return agg

    agg_a = aggregate(runs_a)
    agg_b = aggregate(runs_b)

    all_ids = sorted(set(list(agg_a.keys()) + list(agg_b.keys())))

    # Try importing scipy for t-test
    try:
        from scipy.stats import ttest_ind
        has_scipy = True
    except ImportError:
        print("WARNING: scipy not available; skipping significance tests.")
        print("         Install with: pip install scipy")
        has_scipy = False

    diff_rows = []
    stats_rows = []

    for mid in all_ids:
        vals_a = agg_a.get(mid, [])
        vals_b = agg_b.get(mid, [])

        if not vals_a or not vals_b:
            continue

        mean_a = statistics.mean(vals_a)
        mean_b = statistics.mean(vals_b)
        std_a = statistics.stdev(vals_a) if len(vals_a) > 1 else 0.0
        std_b = statistics.stdev(vals_b) if len(vals_b) > 1 else 0.0

        delta = mean_b - mean_a
        delta_pct = (100.0 * delta / abs(mean_a)) if mean_a != 0 else (0.0 if delta == 0 else float("inf"))

        status = classify_change(mid, mean_a, mean_b, thresholds)

        # Statistical test
        p_value = None
        significant = False
        if has_scipy and len(vals_a) > 1 and len(vals_b) > 1:
            try:
                _, p_value = ttest_ind(vals_a, vals_b, equal_var=False)
                significant = p_value < 0.05
            except Exception:
                pass

        diff_rows.append({
            "metric": mid,
            "val_A": mean_a,
            "val_B": mean_b,
            "delta": delta,
            "delta_pct": delta_pct,
            "status": status,
        })

        stats_rows.append({
            "metric": mid,
            "mean_A": mean_a,
            "std_A": std_a,
            "mean_B": mean_b,
            "std_B": std_b,
            "delta": delta,
            "delta_pct": delta_pct,
            "p_value": p_value if p_value is not None else "N/A",
            "significant": "yes" if significant else "no",
            "status": status,
            "sig_regressed": "YES" if (significant and status == "regressed") else "no",
        })

    return diff_rows, stats_rows

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_ab_diff_csv(rows, ref_a, ref_b, path):
    """Write output/ab_diff.csv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", f"val_A ({ref_a})", f"val_B ({ref_b})", "delta", "delta_pct", "status"])
        for r in rows:
            w.writerow([
                r["metric"],
                f"{r['val_A']:.6f}" if isinstance(r["val_A"], float) else r["val_A"],
                f"{r['val_B']:.6f}" if isinstance(r["val_B"], float) else r["val_B"],
                f"{r['delta']:.6f}" if isinstance(r["delta"], float) else r["delta"],
                f"{r['delta_pct']:.2f}" if isinstance(r["delta_pct"], float) and not math.isinf(r["delta_pct"]) else r["delta_pct"],
                r["status"],
            ])
    print(f"\nWrote: {path}")


def write_ab_stats_csv(stats_rows, ref_a, ref_b, path):
    """Write output/ab_stats.csv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "metric",
            f"mean_A ({ref_a})", f"std_A",
            f"mean_B ({ref_b})", f"std_B",
            "delta", "delta_pct",
            "p_value", "significant", "status", "sig_regressed",
        ])
        for r in stats_rows:
            def fmt(v):
                if isinstance(v, float) and not math.isinf(v):
                    return f"{v:.6f}"
                return str(v)
            w.writerow([
                r["metric"],
                fmt(r["mean_A"]), fmt(r["std_A"]),
                fmt(r["mean_B"]), fmt(r["std_B"]),
                fmt(r["delta"]), f"{r['delta_pct']:.2f}" if isinstance(r["delta_pct"], float) and not math.isinf(r["delta_pct"]) else str(r["delta_pct"]),
                fmt(r["p_value"]),
                r["significant"],
                r["status"],
                r["sig_regressed"],
            ])
    print(f"Wrote: {path}")


def print_summary_table(rows, ref_a, ref_b, stats_rows=None):
    """Print coloured summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"  A/B Comparison: {bold(ref_a)} (A) vs {bold(ref_b)} (B)")
    print(f"{'='*80}")

    # Compute column widths
    hdr = ["Metric", "Val A", "Val B", "Delta", "Delta%", "Status"]
    col_w = [len(h) for h in hdr]

    formatted = []
    for r in rows:
        def fv(v):
            if isinstance(v, float):
                if math.isinf(v):
                    return "inf"
                return f"{v:.4f}"
            return str(v)

        vals = [
            r["metric"],
            fv(r["val_A"]),
            fv(r["val_B"]),
            fv(r["delta"]),
            f"{r['delta_pct']:.1f}%" if isinstance(r["delta_pct"], float) and not math.isinf(r["delta_pct"]) else str(r["delta_pct"]),
            r["status"],
        ]
        formatted.append(vals)
        for i, v in enumerate(vals):
            col_w[i] = max(col_w[i], len(v))

    # Print header
    hdr_line = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(hdr))
    print(f"\n{bold(hdr_line)}")
    print("-" * len(hdr_line))

    # Print rows
    regressions = 0
    improvements = 0
    for vals in formatted:
        status = vals[-1]
        line = "  ".join(v.ljust(col_w[i]) for i, v in enumerate(vals))
        if status == "regressed":
            print(red(line))
            regressions += 1
        elif status == "improved":
            print(green(line))
            improvements += 1
        elif status == "unchanged":
            print(line)
        else:
            print(yellow(line))

    # Summary
    print(f"\n{bold('Summary:')} {green(f'{improvements} improved')}, "
          f"{red(f'{regressions} regressed')}, "
          f"{len(rows) - improvements - regressions} unchanged/other")

    # If stats mode, print significance warnings
    if stats_rows:
        sig_reg = [r for r in stats_rows if r["sig_regressed"] == "YES"]
        if sig_reg:
            print(f"\n{red(bold('SIGNIFICANT REGRESSIONS (p < 0.05):'))}")
            for r in sig_reg:
                pv = r["p_value"]
                pv_str = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
                print(red(f"  {r['metric']}: delta={r['delta']:.4f} p={pv_str}"))

    return regressions

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="A/B comparison of test_regression metrics between two git refs.",
        epilog="Example: python ab_compare.py main feature/my-change --repeat 5",
    )
    parser.add_argument("ref_a", help="Git ref for version A (baseline)")
    parser.add_argument("ref_b", help="Git ref for version B (candidate)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run each version N times (default: 1). Enables statistical analysis.")
    parser.add_argument("--target", default="all",
                        help='Test target/sections (default: "all"). E.g. "1 2 3" or "all".')
    parser.add_argument("--build-dir", default="build",
                        help="CMake build directory (default: build)")
    args = parser.parse_args()

    # Validate
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    # Remember where we are
    original_ref = git_current_ref()
    stashed = False

    print(f"A/B Compare: {bold(args.ref_a)} vs {bold(args.ref_b)}")
    print(f"  Repeat: {args.repeat}, Target: {args.target}, Build dir: {args.build_dir}")
    print(f"  Current ref: {original_ref}")

    # Load thresholds for direction info
    thresholds = load_thresholds()

    runs_a = None
    runs_b = None

    try:
        # Stash dirty state
        stashed = git_stash_if_dirty()

        # --- Collect A ---
        t0 = time.time()
        runs_a = collect_metrics(args.ref_a, args.build_dir, args.target, args.repeat)
        t_a = time.time() - t0

        # --- Collect B ---
        t0 = time.time()
        runs_b = collect_metrics(args.ref_b, args.build_dir, args.target, args.repeat)
        t_b = time.time() - t0

        print(f"\nCollection time: A={t_a:.1f}s, B={t_b:.1f}s")

    except Exception as e:
        print(f"\n{red(bold('ERROR:'))} {e}")
        print("Restoring git state...")
        # Fall through to finally
        if runs_a is None or runs_b is None:
            # Cannot produce comparison
            _restore_git(original_ref, stashed)
            return 1

    finally:
        # Always restore git state
        _restore_git(original_ref, stashed)

    # --- Compare ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.repeat == 1:
        # Single-run comparison
        diff_rows = compute_single_diff(runs_a[0], runs_b[0], thresholds)
        diff_path = os.path.join(OUTPUT_DIR, "ab_diff.csv")
        write_ab_diff_csv(diff_rows, args.ref_a, args.ref_b, diff_path)
        n_regressed = print_summary_table(diff_rows, args.ref_a, args.ref_b)
    else:
        # Multi-run with stats
        diff_rows, stats_rows = compute_stats_diff(runs_a, runs_b, thresholds)
        diff_path = os.path.join(OUTPUT_DIR, "ab_diff.csv")
        stats_path = os.path.join(OUTPUT_DIR, "ab_stats.csv")
        write_ab_diff_csv(diff_rows, args.ref_a, args.ref_b, diff_path)
        write_ab_stats_csv(stats_rows, args.ref_a, args.ref_b, stats_path)
        n_regressed = print_summary_table(diff_rows, args.ref_a, args.ref_b, stats_rows)

        # For --repeat mode, count significant regressions specifically
        sig_reg = [r for r in stats_rows if r["sig_regressed"] == "YES"]
        if sig_reg:
            print(f"\n{red(bold(f'{len(sig_reg)} statistically significant regression(s) detected.'))}")
            return 1

    if n_regressed > 0:
        print(f"\n{red(bold(f'{n_regressed} regression(s) detected.'))}")
        return 1

    print(f"\n{green(bold('No regressions detected.'))}")
    return 0


def _restore_git(original_ref, stashed):
    """Restore git to the original state."""
    try:
        print(f"\nRestoring git state: {original_ref}")
        git_checkout(original_ref)
        if stashed:
            print("  Popping stash...")
            git_stash_pop()
        print("  Git state restored.")
    except Exception as e:
        print(f"  WARNING: Failed to restore git state: {e}")
        print(f"  You may need to manually: git checkout {original_ref}")
        if stashed:
            print(f"  And: git stash pop")


if __name__ == "__main__":
    sys.exit(main())
