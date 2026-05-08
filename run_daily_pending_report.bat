@echo off

cd /d C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST

call .venv\Scripts\activate

python daily_pending_confirm_report.py

python send_pending_emails.py