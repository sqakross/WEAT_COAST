from pathlib import Path
from datetime import datetime
import ast
import sys
import traceback

from flask import (
    Flask,
    Blueprint,
    abort,
    request,
    redirect,
    url_for,
    flash,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
)

ROOT = Path.cwd().resolve()
REPORT_DIR = ROOT / "_patch_reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT = REPORT_DIR / (
    "employee_time_5b3_http_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    + ".txt"
)

log = []
failures = 0

def record(name, passed, details=""):
    global failures
    status = "PASS" if passed else "FAIL"
    log.append(f"{status}: {name}")
    if details:
        log.append("  " + str(details))
    if not passed:
        failures += 1

class TestUser(UserMixin):
    def __init__(self, user_id, role):
        self.id = user_id
        self.role = role

    def get_id(self):
        return str(self.id)

users = {
    "1": TestUser(1, "superadmin"),
    "2": TestUser(2, "admin"),
}

app = Flask("employee_time_5b3_isolated")
app.config.update(
    TESTING=True,
    SECRET_KEY="isolated-test-only-secret",
    SERVER_NAME="localhost",
)

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

bp = Blueprint("inventory", __name__)

# Load only the installed endpoint, not inventory/routes.py.
source = (ROOT / "inventory/routes.py").read_text(
    encoding="utf-8-sig"
)
tree = ast.parse(source)

matches = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "employee_time_tracking_setting"
]

if len(matches) != 1:
    raise SystemExit("STOP: Installed endpoint not found uniquely")

endpoint_ast = matches[0]

module = ast.Module(
    body=[endpoint_ast],
    type_ignores=[],
)

ast.fix_missing_locations(module)

namespace = {
    "inventory_bp": bp,
    "login_required": login_required,
    "current_user": current_user,
    "request": request,
    "abort": abort,
    "flash": flash,
    "redirect": redirect,
    "url_for": url_for,
}

exec(
    compile(module, str(ROOT / "inventory/routes.py"), "exec"),
    namespace,
)

app.register_blueprint(bp)

# Stub only the service boundary.
import services.employee_time_settings as settings

original_verify = settings.verify_csrf_token
original_set = settings.set_tracking_enabled

calls = []

def fake_verify(token, actor_id, target_id):
    calls.append(("csrf", token, actor_id, target_id))
    return (
        token == "valid-token"
        and actor_id == 1
        and target_id == 10
    )

def fake_set(actor, target_id, enabled):
    calls.append(("set", actor.id, target_id, enabled))

settings.verify_csrf_token = fake_verify
settings.set_tracking_enabled = fake_set

def client_as(user_id):
    client = app.test_client()
    with client.session_transaction() as session:
        session["_user_id"] = str(user_id)
        session["_fresh"] = True
    return client

def post(client, token="valid-token", enabled="1", target=10):
    return client.post(
        f"/users/{target}/employee-time-tracking",
        data={
            "employee_time_csrf": token,
            "enabled": enabled,
        },
        follow_redirects=False,
    )

try:
    superadmin = client_as(1)
    admin = client_as(2)

    # 1. Superadmin enables tracking.
    calls.clear()
    response = post(superadmin)

    record(
        "Superadmin accepted",
        response.status_code == 302
        and ("set", 1, 10, True) in calls,
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 2. Superadmin disables tracking.
    calls.clear()
    response = post(superadmin, enabled="0")

    record(
        "Superadmin disable",
        response.status_code == 302
        and ("set", 1, 10, False) in calls,
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 3. Admin must be rejected before service call.
    calls.clear()
    response = post(admin)

    record(
        "Admin forbidden",
        response.status_code == 403
        and not calls,
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 4. Invalid CSRF.
    calls.clear()
    response = post(superadmin, token="invalid")

    record(
        "Invalid CSRF rejected",
        response.status_code == 400
        and not any(call[0] == "set" for call in calls),
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 5. Wrong target CSRF.
    calls.clear()
    response = post(superadmin, target=11)

    record(
        "Wrong target rejected",
        response.status_code == 400
        and not any(call[0] == "set" for call in calls),
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 6. Invalid enabled value.
    calls.clear()
    response = post(superadmin, enabled="false")

    record(
        "Invalid enabled value rejected",
        response.status_code == 400
        and not any(call[0] == "set" for call in calls),
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 7. Missing CSRF.
    calls.clear()
    response = post(superadmin, token="")

    record(
        "Missing CSRF rejected",
        response.status_code == 400
        and not any(call[0] == "set" for call in calls),
        f"HTTP {response.status_code}; calls={calls}",
    )

    # 8. Anonymous access.
    calls.clear()
    anonymous = app.test_client()
    response = post(anonymous)

    record(
        "Anonymous cannot change tracking",
        response.status_code != 302
        and not calls,
        f"HTTP {response.status_code}; calls={calls}",
    )

except Exception:
    failures += 1
    log.append("FAIL: Unexpected test exception")
    log.append(traceback.format_exc())

finally:
    settings.verify_csrf_token = original_verify
    settings.set_tracking_enabled = original_set

    log.append("")
    log.append(f"FAILED CHECKS: {failures}")
    log.append("PRODUCTION DATABASE MODIFIED: NO")
    log.append("APPLICATION SOURCE MODIFIED: NO")

    REPORT.write_text("\n".join(log), encoding="utf-8")

    print(
        "HTTP TEST RESULT:",
        "PASS" if failures == 0 else "FAIL",
    )
    print("Failed checks:", failures)
    print("Report:", REPORT.relative_to(ROOT))

if failures:
    sys.exit(1)
