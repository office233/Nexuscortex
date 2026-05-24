@echo off
REM Build script for CUDA nexus DLL using MSVC toolchain
REM Must be run from VS Developer Command Prompt or with vcvarsall.bat

REM Setup MSVC environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Compile CUDA kernel to object file
nvcc -c -o forward_sparse.obj forward_sparse.cu -arch=sm_75

REM Link into DLL with MSVC linker (resolves CRT symbols correctly)
link /DLL /OUT:cuda_nexus.dll forward_sparse.obj cudart.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64" /NOLOGO

REM Generate import library for MinGW/GCC linker
REM The .lib file from MSVC link is COFF format, we need to generate a .def and then a MinGW .a
dumpbin /EXPORTS cuda_nexus.dll > exports.txt

echo LIBRARY cuda_nexus > cuda_nexus.def
echo EXPORTS >> cuda_nexus.def
for /f "tokens=4" %%a in ('findstr /r "nexus_cuda_" exports.txt') do (
    echo     %%a >> cuda_nexus.def
)

dlltool -d cuda_nexus.def -l libcuda_nexus.a

echo Done! Files created:
echo   cuda_nexus.dll  - CUDA runtime library
echo   libcuda_nexus.a - MinGW import library for CGO
