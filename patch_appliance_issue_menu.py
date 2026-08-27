from pathlib import Path
from datetime import datetime

path = Path("templates/base.html")

if not path.exists():
    raise SystemExit(
        "ERROR: templates/base.html not found"
    )

text = path.read_text(
    encoding="utf-8"
)

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup = Path(
    str(path)
    + f".before_appliance_issue_menu_{stamp}.bak"
)

backup.write_text(
    text,
    encoding="utf-8"
)

print("BACKUP:", backup)


# ============================================================
# Already installed?
# ============================================================

if "NEW APPLIANCE ISSUE" in text:

    print(
        "SKIP: NEW APPLIANCE ISSUE already exists in menu"
    )

    raise SystemExit(0)


# ============================================================
# Insert immediately before MISSING PRICES
# ============================================================

needle = '''                <li>
                  <a class="dropdown-item{% if request.endpoint == 'appliance.receiving_missing_prices' %} active{% endif %}"
                     href="{{ url_for('appliance.receiving_missing_prices') }}">
                    MISSING PRICES
                  </a>
                </li>
'''

replacement = '''                <li>
                  <hr class="dropdown-divider">
                </li>

                <li>
                  <a class="dropdown-item fw-semibold{% if request.endpoint and request.endpoint.startswith('appliance.issue') %} active{% endif %}"
                     href="{{ url_for('appliance.issue_new') }}">
                    NEW APPLIANCE ISSUE
                  </a>
                </li>

                <li>
                  <hr class="dropdown-divider">
                </li>

                <li>
                  <a class="dropdown-item{% if request.endpoint == 'appliance.receiving_missing_prices' %} active{% endif %}"
                     href="{{ url_for('appliance.receiving_missing_prices') }}">
                    MISSING PRICES
                  </a>
                </li>
'''

if needle not in text:

    raise SystemExit(
        "ERROR: MISSING PRICES menu block not found. "
        "No changes made."
    )

text = text.replace(
    needle,
    replacement,
    1,
)

path.write_text(
    text,
    encoding="utf-8"
)

print()
print("=" * 60)
print("APPLIANCE MENU UPDATED")
print("=" * 60)
print("INVENTORY")
print("RECEIVING")
print("NEW APPLIANCE ISSUE")
print("MISSING PRICES")
print("=" * 60)

