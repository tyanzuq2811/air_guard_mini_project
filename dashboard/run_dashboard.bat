@echo off
REM Beijing Air Quality Dashboard Launcher
REM =======================================

echo.
echo ============================================
echo  Beijing Air Quality Dashboard
echo ============================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit chua duoc cai dat!
    echo.
    echo Cai dat dependencies bang lenh:
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if data exists
if not exist "..\data\processed\dataset_for_semi.parquet" (
    echo [WARNING] Khong tim thay du lieu processed!
    echo Vui long chay preprocessing notebook truoc.
    echo.
)

echo Starting dashboard...
echo.
echo Dashboard se mo tai: http://localhost:8501
echo.
echo Nhan Ctrl+C de dung dashboard
echo.

streamlit run app.py

pause
