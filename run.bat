@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Missing .venv\Scripts\python.exe
    echo Run this first:
    echo python -m venv .venv
    echo .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Face Recognition App at http://127.0.0.1:8000/
start "Face Recognition API" cmd /k "cd /d ""%~dp0backend"" && ..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo Open this URL in your browser:
echo http://127.0.0.1:8000/
