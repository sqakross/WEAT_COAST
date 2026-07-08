@echo off
setlocal

set "PROJECT_DIR=C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "APP=%PROJECT_DIR%\app.py"

cd /d "%PROJECT_DIR%" || exit /b 1

set PORT=5000
set DEBUG=false
set USE_SSL=1

chcp 65001 >NUL
set PYTHONUTF8=1
set PYTHONIOENCODING=UTF-8
set PYTHONUNBUFFERED=1

"%PYTHON_EXE%" "%APP%"

exit /b %ERRORLEVEL%