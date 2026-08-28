from pathlib import Path
from datetime import datetime
import re

SERVICE = Path("services/appliance_issue_service.py")
TEMPLATE = Path("templates/appliance_inventory_detail.html")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path):
    dst = Path(str(path) + f".before_loaner_condition_{stamp}.bak")
    dst.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"BACKUP: {dst}")


for path in (SERVICE, TEMPLATE):
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found")
    backup(path)


# ============================================================
# 1. SERVICE
# Replace old single LOANER_RETURN_TO_STOCK transition
# with explicit safe condition choices.
# ============================================================

text = SERVICE.read_text(encoding="utf-8")

pattern = re.compile(
    r'''
            "LOANER_RETURN_TO_STOCK":\s*\{
.*?
            \},\n
''',
    re.DOTALL | re.VERBOSE,
)

replacement = '''            "LOANER_RETURN_TO_STOCK_USED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "used",
            },

            "LOANER_RETURN_TO_STOCK_REPAIRED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "repaired",
            },

            "LOANER_RETURN_TO_STOCK_OPEN_BOX": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "open_box",
            },

            "LOANER_RETURN_TO_STOCK_NEW": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "new",
            },

            "LOANER_RETURN_TO_STOCK_DAMAGED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "damaged",
            },

'''

text, count = pattern.subn(
    replacement,
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "ERROR: existing LOANER_RETURN_TO_STOCK transition "
        "not found exactly once. Nothing written."
    )

SERVICE.write_text(text, encoding="utf-8")

print("OK: service transition replaced with 5 safe choices")


# ============================================================
# 2. TEMPLATE
#
# Existing first patch created:
#   <div class="alert alert-info ...">
#       ...
#       value="LOANER_RETURN_TO_STOCK"
#       ...
#   </div>
#
# Replace that entire LOANER block.
# ============================================================

text = TEMPLATE.read_text(encoding="utf-8")

start_marker = '''          {% if unit.stock_class == 'loaner' %}'''
end_marker = '''          {% endif %}

          <div class="row g-3">'''

start = text.find(start_marker)

if start == -1:
    raise SystemExit(
        "ERROR: LOANER action block start not found."
    )

end = text.find(
    end_marker,
    start,
)

if end == -1:
    raise SystemExit(
        "ERROR: LOANER action block end not found."
    )

new_block = '''          {% if unit.stock_class == 'loaner' %}

            <div class="alert alert-info mb-3">

              <div class="row align-items-end g-3">

                <div class="col-lg-5">

                  <div class="fw-bold">
                    LOANER STOCK
                  </div>

                  <div class="small">
                    Return this appliance from the Loaner fund
                    to regular stock.
                  </div>

                </div>

                <div class="col-lg-7">

                  <form method="post"
                        action="{{ url_for(
                            'appliance.inventory_disposition',
                            unit_id=unit.id
                        ) }}"
                        class="row align-items-end g-2"
                        onsubmit="return confirm(
                          'RETURN THIS LOANER TO REGULAR STOCK?'
                        );">

                    <div class="col-md-7">

                      <label class="form-label small fw-bold mb-1">
                        Condition after return
                      </label>

                      <select name="action"
                              class="form-select form-select-sm">

                        <option value="LOANER_RETURN_TO_STOCK_USED"
                                selected>
                          USED
                        </option>

                        <option value="LOANER_RETURN_TO_STOCK_REPAIRED">
                          REPAIRED
                        </option>

                        <option value="LOANER_RETURN_TO_STOCK_OPEN_BOX">
                          OPEN BOX
                        </option>

                        <option value="LOANER_RETURN_TO_STOCK_NEW">
                          NEW
                        </option>

                        <option value="LOANER_RETURN_TO_STOCK_DAMAGED">
                          DAMAGED
                        </option>

                      </select>

                    </div>

                    <div class="col-md-5">

                      <input type="hidden"
                             name="reason_code"
                             value="LOANER_RETURNED_TO_STOCK">

                      <input type="hidden"
                             name="notes"
                             value="Returned from Loaner fund to regular stock.">

                      <button type="submit"
                              class="btn btn-dark btn-sm w-100">
                        RETURN TO STOCK
                      </button>

                    </div>

                  </form>

                </div>

              </div>

            </div>

          {% endif %}

          <div class="row g-3">'''

text = (
    text[:start]
    + new_block
    + text[end + len(end_marker):]
)

TEMPLATE.write_text(text, encoding="utf-8")

print("OK: Loaner UI replaced with condition selector")


# ============================================================
# 3. SELF CHECK
# ============================================================

service = SERVICE.read_text(encoding="utf-8")
template = TEMPLATE.read_text(encoding="utf-8")

actions = [
    "LOANER_RETURN_TO_STOCK_USED",
    "LOANER_RETURN_TO_STOCK_REPAIRED",
    "LOANER_RETURN_TO_STOCK_OPEN_BOX",
    "LOANER_RETURN_TO_STOCK_NEW",
    "LOANER_RETURN_TO_STOCK_DAMAGED",
]

print()
print("=" * 72)
print("SELF CHECK")
print("=" * 72)

for action in actions:
    print(
        f"{action}:",
        "SERVICE OK" if f'"{action}"' in service else "SERVICE MISSING",
        "|",
        "UI OK" if f'value="{action}"' in template else "UI MISSING",
    )

old_exact = '"LOANER_RETURN_TO_STOCK":'

print()
print(
    "Old transition removed:",
    "YES" if old_exact not in service else "NO"
)

print(
    "Default USED:",
    "YES"
    if '''value="LOANER_RETURN_TO_STOCK_USED"
                                selected''' in template
    else "NO"
)

print()
print("RESULT:")
print("  Status      -> AVAILABLE")
print("  Stock Class -> NEW")
print("  Condition   -> selected by manager")
print("  Default     -> USED")
print()
print("NO MIGRATION")
print("=" * 72)

