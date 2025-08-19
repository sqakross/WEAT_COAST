@echo off
setlocal ENABLEDELAYEDEXPANSION

set "PROJECT_DIR=C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "APP=%PROJECT_DIR%\app.py"
set "LOG_DIR=%PROJECT_DIR%\logs"
set "OUT_LOG=%LOG_DIR%\app-out.log"
set "ERR_LOG=%LOG_DIR%\app-err.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%PROJECT_DIR%" || (echo [%DATE% %TIME%] cd failed>>"%ERR_LOG%" & exit /b 1)

rem отключаем ребилдер чтобы не плодить процессы

set PORT=5000
chcp 65001 >NUL
set PYTHONUTF8=1
set PYTHONIOENCODING=UTF-8
set PYTHONUNBUFFERED=1

echo [%DATE% %TIME%] launching app.py >> "%OUT_LOG%"
"%PYTHON_EXE%" "%APP%" 1>>"%OUT_LOG%" 2>>"%ERR_LOG%"
echo [%DATE% %TIME%] app.py exit %ERRORLEVEL% >> "%OUT_LOG%"
exit /b %ERRORLEVEL%
