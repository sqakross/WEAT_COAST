from pathlib import Path
from datetime import datetime

service_path = Path(
    "services/appliance_issue_service.py"
)

template_path = Path(
    "templates/appliance_inventory_detail.html"
)

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


def backup(path: Path):

    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found"
        )

    dst = Path(
        str(path)
        + f".before_loaner_return_to_stock_{stamp}.bak"
    )

    dst.write_text(
        path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    print(
        "BACKUP:",
        dst
    )


backup(service_path)
backup(template_path)


# ============================================================
# 1. SERVICE TRANSITION
# ============================================================

text = service_path.read_text(
    encoding="utf-8"
)

if '"LOANER_RETURN_TO_STOCK"' not in text:

    marker = '''            "SEND_TO_REPAIR": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_REPAIR,

                "movement":
                    ApplianceIssueService.MOVEMENT_SEND_TO_REPAIR,

                "default_reason":
                    "NEEDS_REPAIR",
            },
'''

    new_transition = marker + '''
            "LOANER_RETURN_TO_STOCK": {
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

'''

    if marker not in text:

        raise SystemExit(
            "ERROR: SEND_TO_REPAIR transition not found. "
            "Nothing changed."
        )

    text = text.replace(
        marker,
        new_transition,
        1,
    )

    print(
        "OK: LOANER_RETURN_TO_STOCK transition added"
    )

else:

    print(
        "SKIP: LOANER_RETURN_TO_STOCK already exists"
    )


service_path.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# 2. DETAIL PAGE ACTION
# ============================================================

text = template_path.read_text(
    encoding="utf-8"
)

if "LOANER_RETURN_TO_STOCK" not in text:

    # --------------------------------------------------------
    # Insert special Loaner action immediately after
    # {% if unit.status == 'available' %}
    # and before existing normal AVAILABLE actions.
    # --------------------------------------------------------

    marker = '''        {% if unit.status == 'available' %}

          <div class="row g-3">
'''

    replacement = '''        {% if unit.status == 'available' %}

          {% if unit.stock_class == 'loaner' %}

            <div class="alert alert-info d-flex justify-content-between align-items-center gap-3">

              <div>

                <div class="fw-bold">
                  LOANER STOCK
                </div>

                <div class="small">
                  This appliance is currently in the Loaner fund.
                  Return it to regular stock when it is no longer
                  needed as a Loaner.
                </div>

              </div>

              <form method="post"
                    action="{{ url_for(
                        'appliance.inventory_disposition',
                        unit_id=unit.id
                    ) }}"
                    class="m-0"
                    onsubmit="return confirm(
                      'RETURN THIS LOANER TO REGULAR STOCK?\\n\\n' +
                      'Stock Class: LOANER → NEW\\n' +
                      'Condition: → USED\\n' +
                      'Status remains AVAILABLE.'
                    );">

                <input type="hidden"
                       name="action"
                       value="LOANER_RETURN_TO_STOCK">

                <input type="hidden"
                       name="reason_code"
                       value="LOANER_RETURNED_TO_STOCK">

                <input type="hidden"
                       name="notes"
                       value="Returned from Loaner fund to regular stock.">

                <button class="btn btn-dark btn-sm">
                  RETURN TO STOCK
                </button>

              </form>

            </div>

          {% endif %}

          <div class="row g-3">
'''

    if marker not in text:

        raise SystemExit(
            "ERROR: AVAILABLE action section not found. "
            "Nothing changed."
        )

    text = text.replace(
        marker,
        replacement,
        1,
    )

    print(
        "OK: Loaner RETURN TO STOCK action added to detail page"
    )

else:

    print(
        "SKIP: Loaner detail action already exists"
    )


template_path.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# SELF CHECK
# ============================================================

service_check = service_path.read_text(
    encoding="utf-8"
)

template_check = template_path.read_text(
    encoding="utf-8"
)

print()
print("=" * 76)
print("LOANER RETURN TO STOCK SELF CHECK")
print("=" * 76)

print(
    "Service transition:",
    service_check.count(
        '"LOANER_RETURN_TO_STOCK"'
    )
)

print(
    "Template action:",
    template_check.count(
        'value="LOANER_RETURN_TO_STOCK"'
    )
)

print()
print("EXPECTED RESULT:")
print("  STATUS      AVAILABLE -> AVAILABLE")
print("  STOCK CLASS LOANER    -> NEW")
print("  CONDITION             -> USED")
print()
print("NO MIGRATION")
print("NO NEW ROUTE")
print("NO DATABASE SCHEMA CHANGE")
print()
print("SUCCESS")
print("=" * 76)

