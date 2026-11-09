@echo off
rem ============================================================
rem  build.bat - compile all 14 numbered steps of the QuadFP-Cascade
rem  C++11 port (g++ 4.9.2 / MinGW, no third-party dependencies).
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo [build] C++11 (g++ -std=c++11 -O2 -static), compiling numbered steps ...
for %%f in (??_*.cpp) do (
    rem 类版本 10_gnn_gcn_class.cpp 需 zlib，由 build_class.bat 单独编译
    echo %%~nf | findstr /c:"_class" >nul
    if errorlevel 1 (
        echo   compiling %%f ...
        g++ -std=c++11 -O2 -static -o "%%~nf.exe" "%%f"
        if errorlevel 1 (
            echo.
            echo [BUILD FAILED] at %%f
            pause
            exit /b 1
        )
    )
)
echo.
echo [build OK] all executables generated next to the sources.
