@echo off
REM === WEST COAST CHIEF REPAIR server launcher ===

REM Переходим в проект
cd /d C:\Users\andrii\PycharmProjects\WEAT_COAST

REM Включаем виртуалку
call .venv\Scripts\activate.bat

REM Делаем папку для логов (на всякий случай)
if not exist logs mkdir logs

REM Запускаем сервер в бесконечном цикле,
REM чтобы если сервер крашнулся, он перезапустился через 5 сек
:loop
echo [%DATE% %TIME%] Starting server... >> logs\server_run.log

REM тут можно запускать Flask dev или waitress / gunicorn-analogue
REM Вариант 1: как у тебя сейчас
python app.py >> logs\server_output.log 2>&1

REM Если вывалился — пишем причину и ждём 5 сек
echo [%DATE% %TIME%] Server crashed or stopped. Restarting in 5 seconds... >> logs\server_run.log
timeout /t 5 /nobreak >nul
goto loop
