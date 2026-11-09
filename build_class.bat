@echo off
rem ============================================================
rem  build_class.bat - compile the class-based GCN step 10
rem  (10_gnn_gcn_class.cpp; needs zlib for real .npz read/write)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set "ZINC=C:\Strawberry\c\include"
set "ZLIB=C:\Strawberry\c\lib"

echo [build] class-based GCN (C++11, thread pool + zlib npz) ...
g++ -std=c++11 -O2 -pthread -I"%ZINC%" -L"%ZLIB%" 10_gnn_gcn_class.cpp -o 10_gnn_gcn_class.exe -lz
if errorlevel 1 (
    echo [BUILD FAILED]
    pause
    exit /b 1
)
echo [build OK] 10_gnn_gcn_class.exe generated.