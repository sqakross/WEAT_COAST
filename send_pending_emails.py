from app import app
from inventory.routes import process_email_queue
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import traceback


def write_log(message: str):
    try:
        log_dir = app.instance_path
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "email_queue.log")

        now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{now_pst.isoformat()} | {message}\n")

    except Exception as e:
        print("LOG ERROR:", e)


if __name__ == "__main__":
    try:
        with app.app_context():
            result = process_email_queue(limit=20)

            msg = f"SUCCESS | sent={result.get('sent')} errors={result.get('errors')} processed={result.get('processed')}"
            print(msg)
            write_log(msg)

    except Exception as e:
        err = f"ERROR | {str(e)}"
        print(err)
        write_log(err)
        write_log(traceback.format_exc())