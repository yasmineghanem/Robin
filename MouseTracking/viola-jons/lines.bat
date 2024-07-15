@echo off
setlocal enabledelayedexpansion
set count=0
for %%f in (*.cpp *.h ) do (
  if not "%%~pf"=="%cd%\stb\" (
        if not "%%~nxf"=="stb*.cpp" (
            if not "%%~nxf"== "stb_image.h" (
                if not "%%~nxf"== "stb_image_write.h" (
                  for /f %%l in ('type "%%f" ^| find /v /c ""') do (
                  set /a count+=%%l
                  )
                )
            )
    )
  )
)
echo Total lines: !count!
endlocal
