@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\TRS001\Documents\workspace\shape_based_matching\build"
cmake --build . --config Release
echo BUILD_EXIT_CODE=%errorlevel%
