from pathlib import Path
from datetime import datetime

routes_path = Path("appliance/routes.py")
template_path = Path("templates/appliance_inventory.html")

if not routes_path.exists():
    raise SystemExit("ERROR: appliance/routes.py not found")

if not template_path.exists():
    raise SystemExit("ERROR: appliance_inventory.html not found")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

routes_backup = Path(
    f"appliance/routes.py.before_inventory_history_{stamp}.bak"
)

template_backup = Path(
    f"templates/appliance_inventory.html.before_history_{stamp}.bak"
)

routes_text = routes_path.read_text(encoding="utf-8")
template_text = template_path.read_text(encoding="utf-8")

routes_backup.write_text(routes_text, encoding="utf-8")
template_backup.write_text(template_text, encoding="utf-8")


# ============================================================
# REPLACE inventory_list()
# ============================================================

start_marker = '''@appliance_bp.get("/inventory")
@login_required
def inventory_list():'''

start = routes_text.find(start_marker)

if start < 0:
    raise SystemExit(
        "ERROR: inventory_list() not found"
    )

# Find next route after inventory_list.
next_route = routes_text.find(
    "\n@appliance_bp.",
    start + len(start_marker),
)

if next_route < 0:
    raise SystemExit(
        "ERROR: could not find route after inventory_list()"
    )


new_inventory_route = r'''@appliance_bp.get("/inventory")
@login_required
def inventory_list():

    from collections import defaultdict

    from sqlalchemy import func, or_

    from models import (
        ApplianceCategory,
        ApplianceIssue,
        ApplianceIssueLine,
        ApplianceUnit,
        Warehouse,
    )

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.view"
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    q = (
        request.args.get("q")
        or ""
    ).strip()

    # IMPORTANT:
    # Inventory is a complete physical registry.
    # Default is ALL, not AVAILABLE.
    status = (
        request.args.get("status")
        or "all"
    ).strip().lower()

    condition = (
        request.args.get("condition")
        or ""
    ).strip().lower()

    raw_warehouse_id = (
        request.args.get("warehouse_id")
        or ""
    ).strip()

    raw_category_id = (
        request.args.get("category_id")
        or ""
    ).strip()

    warehouse_id = None
    category_id = None

    try:
        if raw_warehouse_id:
            warehouse_id = int(
                raw_warehouse_id
            )
    except (TypeError, ValueError):
        warehouse_id = None

    try:
        if raw_category_id:
            category_id = int(
                raw_category_id
            )
    except (TypeError, ValueError):
        category_id = None

    if (
        warehouse_id is not None
        and warehouse_id not in allowed_warehouse_ids
    ):
        warehouse_id = None

    # --------------------------------------------------------
    # Empty access
    # --------------------------------------------------------

    if not allowed_warehouse_ids:

        return render_template(
            "appliance_inventory.html",
            units=[],
            grouped_units=[],
            warehouses=[],
            categories=_active_categories(),
            q=q,
            warehouse_id=None,
            category_id=None,
            status="all",
            condition=condition,
            can_pricing=False,
            total_count=0,
            available_count=0,
            issued_count=0,
            issue_info={},
        )

    # --------------------------------------------------------
    # Base registry query
    # --------------------------------------------------------

    query = (
        ApplianceUnit.query
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            )
        )
    )

    if warehouse_id is not None:
        query = query.filter(
            ApplianceUnit.warehouse_id
            == warehouse_id
        )

    if category_id is not None:
        query = query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if condition:
        query = query.filter(
            ApplianceUnit.condition
            == condition
        )

    if (
        status
        and status != "all"
    ):
        query = query.filter(
            ApplianceUnit.status
            == status
        )

    # --------------------------------------------------------
    # Search
    #
    # Physical fields + warehouse Issue history.
    # --------------------------------------------------------

    if q:

        like = f"%{q}%"

        issue_unit_ids = (
            db.session.query(
                ApplianceIssueLine.appliance_unit_id
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .filter(
                or_(
                    ApplianceIssue.issue_number.ilike(
                        like
                    ),
                    ApplianceIssue.work_order_number.ilike(
                        like
                    ),
                )
            )
        )

        # Technician username needs User.
        from models import User

        issue_unit_ids_by_tech = (
            db.session.query(
                ApplianceIssueLine.appliance_unit_id
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .outerjoin(
                User,
                User.id
                == ApplianceIssue.technician_id,
            )
            .filter(
                User.username.ilike(
                    like
                )
            )
        )

        query = query.filter(
            or_(
                ApplianceUnit.inventory_number.ilike(
                    like
                ),
                ApplianceUnit.serial_number.ilike(
                    like
                ),
                ApplianceUnit.model_number.ilike(
                    like
                ),
                ApplianceUnit.brand.ilike(
                    like
                ),
                ApplianceUnit.description.ilike(
                    like
                ),
                ApplianceUnit.current_work_order_number.ilike(
                    like
                ),
                ApplianceUnit.id.in_(
                    issue_unit_ids
                ),
                ApplianceUnit.id.in_(
                    issue_unit_ids_by_tech
                ),
            )
        )

    # --------------------------------------------------------
    # Counts across complete accessible registry.
    #
    # Counts intentionally ignore Status filter so user always
    # knows total / available / issued inventory.
    # Warehouse/category/condition filters are respected.
    # --------------------------------------------------------

    count_query = (
        db.session.query(
            ApplianceUnit.status,
            func.count(
                ApplianceUnit.id
            ),
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            )
        )
    )

    if warehouse_id is not None:
        count_query = count_query.filter(
            ApplianceUnit.warehouse_id
            == warehouse_id
        )

    if category_id is not None:
        count_query = count_query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if condition:
        count_query = count_query.filter(
            ApplianceUnit.condition
            == condition
        )

    count_rows = (
        count_query
        .group_by(
            ApplianceUnit.status
        )
        .all()
    )

    status_counts = {
        (row[0] or "").lower():
            int(row[1] or 0)
        for row in count_rows
    }

    registry_total_count = sum(
        status_counts.values()
    )

    available_count = status_counts.get(
        "available",
        0,
    )

    issued_count = status_counts.get(
        "issued",
        0,
    )

    # --------------------------------------------------------
    # Load visible rows
    # --------------------------------------------------------

    units = (
        query
        .order_by(
            ApplianceUnit.category_id.asc(),
            ApplianceUnit.brand.asc(),
            ApplianceUnit.model_number.asc(),
            ApplianceUnit.inventory_number.asc(),
        )
        .limit(500)
        .all()
    )

    visible_unit_ids = [
        unit.id
        for unit in units
    ]

    # --------------------------------------------------------
    # Latest Issue info for each physical unit
    #
    # We do NOT store this directly on ApplianceUnit because
    # Issue/Movement history is authoritative.
    # --------------------------------------------------------

    issue_info = {}

    if visible_unit_ids:

        issue_rows = (
            db.session.query(
                ApplianceIssueLine,
                ApplianceIssue,
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .filter(
                ApplianceIssueLine.appliance_unit_id.in_(
                    visible_unit_ids
                )
            )
            .order_by(
                ApplianceIssueLine.appliance_unit_id.asc(),
                ApplianceIssue.issued_at.desc(),
                ApplianceIssue.id.desc(),
            )
            .all()
        )

        for line, issue in issue_rows:

            unit_id = int(
                line.appliance_unit_id
            )

            # First row is newest because of ORDER BY.
            if unit_id in issue_info:
                continue

            issue_info[unit_id] = {
                "issue_id":
                    issue.id,

                "issue_number":
                    issue.issue_number,

                "technician":
                    (
                        issue.technician.username
                        if issue.technician
                        else ""
                    ),

                "work_order_number":
                    (
                        line.current_work_order_number
                        or issue.work_order_number
                        or ""
                    ),

                "issued_at":
                    issue.issued_at_local,
            }

    # --------------------------------------------------------
    # Group visible units by category.
    # Keep existing UI grouping.
    # --------------------------------------------------------

    grouped_map = defaultdict(list)

    for unit in units:

        grouped_map[
            unit.category_id
        ].append(
            unit
        )

    grouped_units = []

    categories_by_id = {
        category.id: category
        for category in _active_categories()
    }

    for category_id_key, rows in grouped_map.items():

        category = categories_by_id.get(
            category_id_key
        )

        if category is None and rows:
            category = rows[0].category

        grouped_units.append(
            {
                "category": category,
                "units": rows,
            }
        )

    grouped_units.sort(
        key=lambda group: (
            (
                group["category"].sort_order
                if group["category"] is not None
                else 999999
            ),
            (
                group["category"].name
                if group["category"] is not None
                else ""
            ),
        )
    )

    # --------------------------------------------------------
    # Visible warehouses
    # --------------------------------------------------------

    warehouses = (
        Warehouse.query
        .filter(
            Warehouse.id.in_(
                allowed_warehouse_ids
            ),
            Warehouse.is_active.is_(True),
        )
        .order_by(
            Warehouse.code.asc()
        )
        .all()
    )

    # --------------------------------------------------------
    # Pricing permission
    # --------------------------------------------------------

    can_pricing = any(
        AccessControlService.can(
            current_user,
            "appliance.pricing",
            warehouse_id=warehouse_id_value,
        )
        for warehouse_id_value
        in allowed_warehouse_ids
    )

    return render_template(
        "appliance_inventory.html",

        units=units,
        grouped_units=grouped_units,

        warehouses=warehouses,
        categories=_active_categories(),

        q=q,
        warehouse_id=warehouse_id,
        category_id=category_id,
        status=status,
        condition=condition,

        can_pricing=can_pricing,

        # total_count = complete filtered registry,
        # NOT only currently visible status.
        total_count=registry_total_count,
        available_count=available_count,
        issued_count=issued_count,

        issue_info=issue_info,
    )


'''

routes_text = (
    routes_text[:start]
    + new_inventory_route
    + routes_text[next_route:]
)

routes_path.write_text(
    routes_text,
    encoding="utf-8",
)


# ============================================================
# FULL TEMPLATE REPLACEMENT
# ============================================================

new_template = r'''{% extends "base.html" %}
{% block content %}

<style>
  .appliance-inventory-page {
    width:calc(100vw - 40px);
    max-width:1800px;
    margin-left:50%;
    transform:translateX(-50%);
  }

  .inventory-toolbar {
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    justify-content:space-between;
    gap:.75rem;
    margin-bottom:1rem;
  }

  .inventory-card {
    background:#fff;
    border:1px solid #d9dee5;
    border-radius:.55rem;
    box-shadow:0 .15rem .45rem rgba(0,0,0,.07);
  }

  .inventory-table {
    font-size:.76rem;
  }

  .inventory-table th,
  .inventory-table td {
    vertical-align:middle;
    white-space:nowrap;
  }

  .inventory-table tbody tr.unit-row {
    cursor:pointer;
  }

  .inventory-table tbody tr.unit-row:hover {
    background:#f4f8fc;
  }

  .inventory-number {
    font-weight:700;
    color:#0d6efd;
  }

  .serial-number {
    font-weight:600;
  }

  .status-badge {
    font-size:.66rem;
    letter-spacing:.03em;
  }

  .filter-label {
    font-size:.68rem;
    font-weight:600;
    color:#6c757d;
    text-transform:uppercase;
    margin-bottom:.2rem;
  }

  .missing-cost {
    background:#fff3cd;
  }

  .inventory-category-row td {
    background:#e9ecef !important;
    border-top:2px solid #adb5bd !important;
    border-bottom:1px solid #adb5bd !important;
    font-size:.76rem;
    font-weight:700;
  }

  .inventory-category-name {
    font-size:.78rem;
    letter-spacing:.04em;
  }

  .inventory-category-count {
    font-size:.65rem;
  }

  .inventory-stat {
    display:inline-flex;
    align-items:center;
    gap:.3rem;
    padding:.28rem .5rem;
    border:1px solid #d7dce2;
    border-radius:.35rem;
    background:#fff;
    font-size:.72rem;
  }

  .inventory-stat strong {
    font-size:.8rem;
  }

  .issue-link {
    font-weight:700;
    text-decoration:none;
  }

  .issue-link:hover {
    text-decoration:underline;
  }

  .issued-row {
    background:#f5f9ff;
  }

  .wo-value {
    font-weight:700;
  }
</style>


<div class="appliance-inventory-page">

  <div class="inventory-toolbar">

    <div>

      <h3 class="mb-1">
        Appliance Inventory
      </h3>

      <div class="small text-muted">
        Complete serialized appliance registry
      </div>

    </div>


    <div class="d-flex flex-wrap gap-2 align-items-center">

      <span class="inventory-stat">
        TOTAL
        <strong>
          {{ total_count }}
        </strong>
      </span>

      <span class="inventory-stat">
        AVAILABLE
        <strong class="text-success">
          {{ available_count }}
        </strong>
      </span>

      <span class="inventory-stat">
        ISSUED
        <strong class="text-primary">
          {{ issued_count }}
        </strong>
      </span>

      <a class="btn btn-primary btn-sm"
         href="{{ url_for('appliance.issue_new') }}">
        + New Issue
      </a>

      <a class="btn btn-outline-primary btn-sm"
         href="{{ url_for('appliance.receiving_list') }}">
        Receiving
      </a>

    </div>

  </div>


  <!-- ======================================================
       FILTERS
       ====================================================== -->

  <div class="inventory-card mb-3">

    <div class="p-3">

      <form method="get"
            action="{{ url_for('appliance.inventory_list') }}">

        <div class="row g-2">


          <div class="col-xl-4 col-md-6">

            <div class="filter-label">
              Search
            </div>

            <input type="search"
                   class="form-control form-control-sm"
                   name="q"
                   value="{{ q or '' }}"
                   placeholder="AP # / SERIAL / MODEL / BRAND / AIS # / TECH / W/O">

          </div>


          <div class="col-xl-2 col-md-3">

            <div class="filter-label">
              Warehouse
            </div>

            <select class="form-select form-select-sm"
                    name="warehouse_id">

              <option value="">
                ALL
              </option>

              {% for warehouse in warehouses %}

                <option value="{{ warehouse.id }}"
                        {% if warehouse_id == warehouse.id %}selected{% endif %}>
                  {{ warehouse.code|upper }}
                </option>

              {% endfor %}

            </select>

          </div>


          <div class="col-xl-2 col-md-3">

            <div class="filter-label">
              Appliance
            </div>

            <select class="form-select form-select-sm"
                    name="category_id">

              <option value="">
                ALL
              </option>

              {% for category in categories %}

                <option value="{{ category.id }}"
                        {% if category_id == category.id %}selected{% endif %}>
                  {{ category.name|upper }}
                </option>

              {% endfor %}

            </select>

          </div>


          <div class="col-xl-2 col-md-3">

            <div class="filter-label">
              Status
            </div>

            <select class="form-select form-select-sm"
                    name="status">

              <option value="all"
                      {% if status == 'all' %}selected{% endif %}>
                ALL
              </option>

              <option value="available"
                      {% if status == 'available' %}selected{% endif %}>
                AVAILABLE
              </option>

              <option value="issued"
                      {% if status == 'issued' %}selected{% endif %}>
                ISSUED
              </option>

              <option value="reserved"
                      {% if status == 'reserved' %}selected{% endif %}>
                RESERVED
              </option>

              <option value="transferred"
                      {% if status == 'transferred' %}selected{% endif %}>
                TRANSFERRED
              </option>

              <option value="vendor_return"
                      {% if status == 'vendor_return' %}selected{% endif %}>
                VENDOR RETURN
              </option>

              <option value="sold"
                      {% if status == 'sold' %}selected{% endif %}>
                SOLD
              </option>

              <option value="written_off"
                      {% if status == 'written_off' %}selected{% endif %}>
                WRITTEN OFF
              </option>

            </select>

          </div>


          <div class="col-xl-2 col-md-3">

            <div class="filter-label">
              Condition
            </div>

            <select class="form-select form-select-sm"
                    name="condition">

              <option value="">
                ALL
              </option>

              <option value="new"
                      {% if condition == 'new' %}selected{% endif %}>
                NEW
              </option>

              <option value="used"
                      {% if condition == 'used' %}selected{% endif %}>
                USED
              </option>

              <option value="open_box"
                      {% if condition == 'open_box' %}selected{% endif %}>
                OPEN BOX
              </option>

              <option value="damaged"
                      {% if condition == 'damaged' %}selected{% endif %}>
                DAMAGED
              </option>

            </select>

          </div>

        </div>


        <div class="d-flex gap-2 mt-3">

          <button type="submit"
                  class="btn btn-primary btn-sm">
            Search
          </button>

          <a class="btn btn-outline-secondary btn-sm"
             href="{{ url_for('appliance.inventory_list') }}">
            Clear
          </a>

        </div>

      </form>

    </div>

  </div>


  <!-- ======================================================
       REGISTRY
       ====================================================== -->

  <div class="inventory-card">

    <div class="table-responsive">

      <table class="table table-sm table-hover inventory-table mb-0">

        <thead class="table-light">

          <tr>

            <th>AP #</th>
            <th>APPLIANCE</th>
            <th>BRAND</th>
            <th>MODEL</th>
            <th>SERIAL</th>
            <th>WAREHOUSE</th>

            {% if can_pricing %}
              <th class="text-end">COST</th>
              <th class="text-end">SELL</th>
            {% endif %}

            <th>STATUS</th>

            <th>ISSUE #</th>
            <th>TECHNICIAN</th>
            <th>W/O #</th>
            <th>ISSUED</th>

            <th>RECEIVED</th>

          </tr>

        </thead>


        <tbody>

          {% for group in grouped_units %}

            {% set column_count = 14 if can_pricing else 12 %}

            <tr class="inventory-category-row">

              <td colspan="{{ column_count }}"
                  class="py-2">

                <span class="inventory-category-name">
                  {{ group.category.name|upper if group.category else 'OTHER' }}
                </span>

                <span class="badge bg-dark ms-2 inventory-category-count">

                  {{ group.units|length }}

                  UNIT{% if group.units|length != 1 %}S{% endif %}

                </span>

              </td>

            </tr>


            {% for unit in group.units %}

              {% set issued = issue_info.get(unit.id) %}

              <tr class="unit-row {% if unit.status == 'issued' %}issued-row{% endif %}"
                  onclick="window.location.href='{{ url_for('appliance.inventory_detail', unit_id=unit.id) }}'">


                <td>

                  <span class="inventory-number">
                    {{ unit.inventory_number }}
                  </span>

                </td>


                <td class="fw-semibold">

                  {{ unit.category.name|upper if unit.category else '—' }}

                </td>


                <td>
                  {{ (unit.brand or '—')|upper }}
                </td>


                <td>
                  {{ (unit.model_number or '—')|upper }}
                </td>


                <td>

                  <span class="serial-number">
                    {{ (unit.serial_number or '—')|upper }}
                  </span>

                </td>


                <td>
                  {{ unit.warehouse.code|upper if unit.warehouse else '—' }}
                </td>


                {% if can_pricing %}

                  <td class="text-end {% if unit.unit_cost is none %}missing-cost{% endif %}">

                    {% if unit.unit_cost is not none %}

                      ${{ '%.2f'|format(unit.unit_cost) }}

                    {% else %}

                      MISSING

                    {% endif %}

                  </td>


                  <td class="text-end">

                    {% if unit.selling_price is not none %}

                      ${{ '%.2f'|format(unit.selling_price) }}

                    {% else %}

                      —

                    {% endif %}

                  </td>

                {% endif %}


                <td>

                  {% if unit.status == 'available' %}

                    <span class="badge bg-success status-badge">
                      AVAILABLE
                    </span>

                  {% elif unit.status == 'issued' %}

                    <span class="badge bg-primary status-badge">
                      ISSUED
                    </span>

                  {% elif unit.status == 'reserved' %}

                    <span class="badge bg-warning text-dark status-badge">
                      RESERVED
                    </span>

                  {% elif unit.status == 'sold' %}

                    <span class="badge bg-dark status-badge">
                      SOLD
                    </span>

                  {% elif unit.status == 'written_off' %}

                    <span class="badge bg-danger status-badge">
                      WRITTEN OFF
                    </span>

                  {% else %}

                    <span class="badge bg-secondary status-badge">

                      {{ (unit.status or '')|replace('_', ' ')|upper }}

                    </span>

                  {% endif %}

                </td>


                <!-- ISSUE # -->

                <td>

                  {% if issued %}

                    <a class="issue-link"
                       href="{{ url_for(
                           'appliance.issue_detail',
                           issue_id=issued.issue_id
                       ) }}"
                       onclick="event.stopPropagation();">

                      {{ issued.issue_number }}

                    </a>

                  {% else %}

                    —

                  {% endif %}

                </td>


                <!-- TECHNICIAN -->

                <td>

                  {% if issued and issued.technician %}

                    {{ issued.technician|upper }}

                  {% else %}

                    —

                  {% endif %}

                </td>


                <!-- W/O -->

                <td class="wo-value">

                  {% if issued and issued.work_order_number %}

                    {{ issued.work_order_number|upper }}

                  {% elif unit.current_work_order_number %}

                    {{ unit.current_work_order_number|upper }}

                  {% else %}

                    —

                  {% endif %}

                </td>


                <!-- ISSUED DATE -->

                <td class="text-muted small">

                  {% if issued and issued.issued_at %}

                    {{ issued.issued_at.strftime('%m/%d/%Y') }}

                  {% else %}

                    —

                  {% endif %}

                </td>


                <!-- RECEIVED -->

                <td class="text-muted small">

                  {% if unit.created_at_local %}

                    {{ unit.created_at_local.strftime('%m/%d/%Y') }}

                  {% else %}

                    —

                  {% endif %}

                </td>

              </tr>

            {% endfor %}


          {% else %}

            <tr>

              <td colspan="{{ 14 if can_pricing else 12 }}"
                  class="text-center text-muted py-5">

                No appliance units found.

              </td>

            </tr>

          {% endfor %}

        </tbody>

      </table>

    </div>

  </div>


  {% if units|length >= 500 %}

    <div class="alert alert-warning mt-3 py-2">

      Showing first 500 matching units.
      Use Search or filters to narrow the registry.

    </div>

  {% endif %}

</div>

{% endblock %}
'''

template_path.write_text(
    new_template,
    encoding="utf-8",
)


print("=" * 70)
print("APPLIANCE INVENTORY HISTORY PATCH")
print("=" * 70)
print("OK: appliance/routes.py updated")
print("OK: appliance_inventory.html replaced")
print()
print("Default status: ALL")
print("Added:")
print("  TOTAL / AVAILABLE / ISSUED counters")
print("  ISSUE #")
print("  TECHNICIAN")
print("  W/O #")
print("  ISSUED DATE")
print("  AIS hyperlink")
print("  AIS / TECH / W/O search")
print()
print("Backups:")
print(" ", routes_backup)
print(" ", template_backup)
print("=" * 70)
