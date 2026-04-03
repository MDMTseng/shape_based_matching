/// @file rigid_solve.h
/// @brief Shared 3x3 LDL^T solver for rigid-body least-squares problems.
///
/// Used by icp_refine.cpp (edge-based ICP) and available for any module
/// that needs to solve a 3x3 symmetric positive-definite normal-equation
/// system (e.g. [theta, tx, ty]).

#ifndef RIGID_SOLVE_H
#define RIGID_SOLVE_H

#include <cmath>

/// Solve A*x = b where A is a 3x3 symmetric positive-definite matrix,
/// using LDL^T decomposition.
///
/// @param A         3x3 SPD matrix (only upper triangle needs to be filled;
///                  lower triangle is read as A[i][j] with i>j).
/// @param b         right-hand side vector (length 3).
/// @param x         output solution vector (length 3).
/// @param epsilon   singularity threshold for diagonal pivots.
/// @return true on success, false if a near-singular pivot is detected.
static inline bool solve3x3_ldl(const float A[3][3], const float b[3],
                                float x[3], float epsilon = 1e-10f) {
    float L[3][3] = {};
    float D[3] = {};

    // Row 0
    D[0] = A[0][0];
    if (std::abs(D[0]) < epsilon) return false;
    L[0][0] = 1;

    // Row 1
    L[1][0] = A[1][0] / D[0];
    D[1] = A[1][1] - L[1][0] * L[1][0] * D[0];
    if (std::abs(D[1]) < epsilon) return false;
    L[1][1] = 1;

    // Row 2
    L[2][0] = A[2][0] / D[0];
    L[2][1] = (A[2][1] - L[2][0] * L[1][0] * D[0]) / D[1];
    D[2] = A[2][2] - L[2][0] * L[2][0] * D[0] - L[2][1] * L[2][1] * D[1];
    if (std::abs(D[2]) < epsilon) return false;
    L[2][2] = 1;

    // Forward substitution: L*y = b
    float y[3];
    y[0] = b[0];
    y[1] = b[1] - L[1][0] * y[0];
    y[2] = b[2] - L[2][0] * y[0] - L[2][1] * y[1];

    // Diagonal: D*z = y
    float z[3];
    z[0] = y[0] / D[0];
    z[1] = y[1] / D[1];
    z[2] = y[2] / D[2];

    // Back substitution: L^T * x = z
    x[2] = z[2];
    x[1] = z[1] - L[2][1] * x[2];
    x[0] = z[0] - L[1][0] * x[1] - L[2][0] * x[2];

    return true;
}

#endif // RIGID_SOLVE_H
