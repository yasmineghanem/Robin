@echo off
REM Change the source and destination paths as needed
set source_path=C:\TempDesktop\fourth_year\GP\ViolaAndJones\CudaRuntime1\x64\Debug\CudaRuntime1.exe
set dest_path=C:\TempDesktop\fourth_year\GP\ViolaAndJones\CudaRuntime1\CudaRuntime1\

REM Copy the file
copy "%source_path%" "%dest_path%"

REM Print a message indicating the operation is complete
echo File copied to %dest_path%

CudaRuntime1.exe