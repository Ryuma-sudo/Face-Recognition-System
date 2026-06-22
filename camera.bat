@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" client\camera_client.py
