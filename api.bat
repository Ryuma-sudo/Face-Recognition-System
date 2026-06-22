@echo off
setlocal
cd /d "%~dp0backend"
echo Face Recognition UI: http://127.0.0.1:8000/
echo API docs: http://127.0.0.1:8000/docs
..\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
