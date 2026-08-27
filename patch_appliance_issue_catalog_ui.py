from pathlib import Path
from datetime import datetime
import re


# ============================================================
# FILES
# ============================================================

routes_path = Path("appliance/routes.py")
template_path = Path("templates/appliance_issue_new.html")

if not routes_path.exists():
    raise SystemExit("ERROR: appliance/routes.py not found")

if not template_path.exists():
    raise SystemExit("ERROR: templates/appliance_issue_new.html not found")


stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

routes_backup = Path(
    f"appliance/routes.py.before_issue_catalog_{stamp}.bak"
)

template_backup = Path(
    f"templates/appliance_issue_new.html.before_catalog_{stamp}.bak"
)

routes_text = routes_path.read_text(
    encoding="utf-8"
)

template_text = template_path.read_text(
    encoding="utf-8"
)

routes_backup.write_text(
    routes_text,
    encoding="utf-8"
)

template_backup.write_text(
    template_text,
    encoding="utf-8"
)


# ============================================================
# REPLACE APPLIANCE SEARCH ENDPOINT
# ============================================================

start_marker = '''@appliance_bp.get(
    "/issues/search-appliances"
)
@login_required
def issue_search_appliances():'''

end_marker = '''# ============================================================
# Search Work Orders
# ============================================================'''

start = routes_text.find(start_marker)

if start < 0:
    raise SystemExit(
        "ERROR: issue_search_appliances route not found"
    )

end = routes_text.find(
    end_marker,
    start,
)

if end < 0:
    raise SystemExit(
        "ERROR: Search Work Orders marker not found"
    )


new_route = r'''@appliance_bp.get(
    "/issues/search-appliances"
)
@login_required
def issue_search_appliances():
    """
    AVAILABLE appliance catalog for warehouse Issue UI.

    Supports:
        warehouse_id
        category_id
        brand
        q

    Response includes:
        category counts
        brands
        filtered physical appliance units

    NO pricing is returned.
    """

    from sqlalchemy import func, or_

    from models import (
        ApplianceCategory,
        ApplianceUnit,
    )

    # --------------------------------------------------------
    # Warehouse
    # --------------------------------------------------------

    try:
        warehouse_id = int(
            request.args.get("warehouse_id")
            or 0
        )
    except (TypeError, ValueError):
        warehouse_id = 0

    if warehouse_id <= 0:
        return jsonify(
            {
                "ok": True,
                "categories": [],
                "brands": [],
                "items": [],
                "total_available": 0,
            }
        )

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=warehouse_id,
    ):
        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    try:
        category_id = int(
            request.args.get("category_id")
            or 0
        )
    except (TypeError, ValueError):
        category_id = 0

    brand = (
        request.args.get("brand")
        or ""
    ).strip().upper()

    q = (
        request.args.get("q")
        or ""
    ).strip()

    # --------------------------------------------------------
    # Category counters
    #
    # Always based on full AVAILABLE inventory in warehouse,
    # independent of currently selected category/brand/search.
    # --------------------------------------------------------

    category_rows = (
        db.session.query(
            ApplianceCategory.id,
            ApplianceCategory.name,
            func.count(
                ApplianceUnit.id
            ).label("unit_count"),
        )
        .join(
            ApplianceUnit,
            ApplianceUnit.category_id
            == ApplianceCategory.id,
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
        )
        .group_by(
            ApplianceCategory.id,
            ApplianceCategory.name,
        )
        .order_by(
            ApplianceCategory.name.asc()
        )
        .all()
    )

    categories = [
        {
            "id": row.id,
            "name": (
                row.name
                or ""
            ).upper(),
            "count": int(
                row.unit_count
                or 0
            ),
        }
        for row in category_rows
    ]

    total_available = sum(
        row["count"]
        for row in categories
    )

    # --------------------------------------------------------
    # Brands
    #
    # Brand list follows selected category, so after clicking
    # REFRIGERATOR we only show refrigerator brands.
    # --------------------------------------------------------

    brand_query = (
        db.session.query(
            ApplianceUnit.brand
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
            ApplianceUnit.brand.isnot(None),
            func.trim(
                ApplianceUnit.brand
            ) != "",
        )
    )

    if category_id > 0:
        brand_query = brand_query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    brand_rows = (
        brand_query
        .distinct()
        .order_by(
            ApplianceUnit.brand.asc()
        )
        .all()
    )

    brands = sorted(
        {
            (
                row[0]
                or ""
            ).strip().upper()
            for row in brand_rows
            if (
                row[0]
                or ""
            ).strip()
        }
    )

    # --------------------------------------------------------
    # Physical AVAILABLE units
    # --------------------------------------------------------

    query = (
        ApplianceUnit.query
        .join(
            ApplianceCategory,
            ApplianceUnit.category_id
            == ApplianceCategory.id,
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
        )
    )

    if category_id > 0:
        query = query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if brand:
        query = query.filter(
            func.upper(
                func.trim(
                    ApplianceUnit.brand
                )
            )
            == brand
        )

    if q:
        like = f"%{q}%"

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
                ApplianceCategory.name.ilike(
                    like
                ),
            )
        )

    units = (
        query
        .order_by(
            ApplianceCategory.name.asc(),
            ApplianceUnit.brand.asc(),
            ApplianceUnit.model_number.asc(),
            ApplianceUnit.serial_number.asc(),
            ApplianceUnit.inventory_number.asc(),
        )
        .limit(100)
        .all()
    )

    items = []

    for unit in units:

        category_name = (
            unit.category.name
            if unit.category
            else ""
        )

        size_text = ""

        if unit.size_value is not None:
            size_number = f"{unit.size_value:g}"

            size_text = (
                f"{size_number} "
                f"{unit.size_unit or ''}"
            ).strip()

        items.append(
            {
                "id": unit.id,

                "inventory_number": (
                    unit.inventory_number
                    or ""
                ).upper(),

                "appliance_type": (
                    category_name
                    or ""
                ).upper(),

                "category_id":
                    unit.category_id,

                "brand": (
                    unit.brand
                    or ""
                ).upper(),

                "model": (
                    unit.model_number
                    or ""
                ).upper(),

                "serial": (
                    unit.serial_number
                    or ""
                ).upper(),

                "size":
                    size_text.upper(),

                "condition": (
                    unit.condition
                    or ""
                ).replace(
                    "_",
                    " ",
                ).upper(),
            }
        )

    return jsonify(
        {
            "ok": True,
            "categories": categories,
            "brands": brands,
            "items": items,
            "total_available":
                total_available,
        }
    )


'''


routes_text = (
    routes_text[:start]
    + new_route
    + routes_text[end:]
)

routes_path.write_text(
    routes_text,
    encoding="utf-8",
)


# ============================================================
# NEW TEMPLATE
# ============================================================

new_template = r'''{% extends "base.html" %}
{% block content %}

<style>
  .issue-page {
    width:min(1500px, calc(100vw - 34px));
    margin:0 auto;
  }

  .issue-card {
    background:#fff;
    border:1px solid #d9dee5;
    border-radius:.55rem;
    box-shadow:0 .15rem .45rem rgba(0,0,0,.07);
  }

  .issue-label {
    font-size:.68rem;
    font-weight:700;
    color:#6c757d;
    text-transform:uppercase;
    margin-bottom:.25rem;
  }

  .issue-page input[type="text"],
  .issue-page input[type="search"],
  .issue-page textarea {
    text-transform:uppercase;
  }

  .search-box {
    position:relative;
  }

  .search-results {
    position:absolute;
    top:100%;
    left:0;
    right:0;
    z-index:1100;
    background:#fff;
    border:1px solid #ced4da;
    border-radius:.35rem;
    box-shadow:0 .35rem .8rem rgba(0,0,0,.14);
    max-height:330px;
    overflow-y:auto;
  }

  .search-result {
    padding:.55rem .7rem;
    border-bottom:1px solid #eceff2;
    cursor:pointer;
  }

  .search-result:hover {
    background:#edf5ff;
  }

  .wo-selected {
    background:#e9f7ef;
    border:1px solid #badbcc;
    border-radius:.35rem;
    padding:.5rem .65rem;
    margin-top:.4rem;
  }

  /* =======================================================
     CATEGORY CARDS
     ======================================================= */

  .category-strip {
    display:flex;
    flex-wrap:wrap;
    gap:.5rem;
  }

  .category-btn {
    min-width:145px;
    padding:.65rem .8rem;
    background:#fff;
    border:1px solid #cfd6de;
    border-radius:.45rem;
    cursor:pointer;
    text-align:left;
    transition:.12s ease;
  }

  .category-btn:hover {
    border-color:#0d6efd;
    background:#f7fbff;
  }

  .category-btn.active {
    border-color:#0d6efd;
    background:#eaf3ff;
    box-shadow:0 0 0 1px #0d6efd inset;
  }

  .category-name {
    font-size:.78rem;
    font-weight:800;
  }

  .category-count {
    font-size:.7rem;
    color:#6c757d;
  }

  /* =======================================================
     AVAILABLE TABLE
     ======================================================= */

  .available-table,
  .selected-table {
    font-size:.78rem;
  }

  .available-table th,
  .available-table td,
  .selected-table th,
  .selected-table td {
    vertical-align:middle;
    white-space:nowrap;
  }

  .available-table tbody tr {
    cursor:default;
  }

  .available-table tbody tr:hover {
    background:#f6f9fc;
  }

  .ap-number {
    font-weight:700;
    color:#0d6efd;
  }

  .inventory-empty {
    text-align:center;
    padding:2.2rem 1rem !important;
    color:#6c757d;
  }

  .selected-row {
    background:#f8fbff;
  }

  .summary-count {
    font-size:1rem;
    font-weight:800;
  }

  .availability-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:1rem;
  }

  .loading-row {
    text-align:center;
    padding:2rem !important;
    color:#6c757d;
  }

  @media (max-width: 992px) {
    .category-btn {
      min-width:125px;
    }
  }
</style>


<div class="issue-page">

  <div class="d-flex justify-content-between align-items-start mb-3">

    <div>
      <h3 class="mb-1">
        New Appliance Issue
      </h3>

      <div class="small text-muted">
        Issue appliances to technician for installation
      </div>
    </div>

    <a class="btn btn-outline-secondary btn-sm"
       href="{{ url_for('appliance.inventory_list') }}">
      ← Inventory
    </a>

  </div>


  <form method="post"
        id="issueForm"
        autocomplete="off">

    <!-- =====================================================
         HEADER
         ===================================================== -->

    <div class="issue-card mb-3">
      <div class="p-3">

        <div class="row g-3">

          <div class="col-lg-3">

            <div class="issue-label">
              Warehouse *
            </div>

            <select class="form-select"
                    name="warehouse_id"
                    id="warehouseSelect"
                    required>

              {% for warehouse in warehouses %}

                <option value="{{ warehouse.id }}"
                        {% if warehouse.id == default_warehouse_id %}selected{% endif %}>
                  {{ warehouse.code|upper }}
                  — {{ warehouse.name|upper }}
                </option>

              {% endfor %}

            </select>

          </div>


          <div class="col-lg-3">

            <div class="issue-label">
              Technician / Employee *
            </div>

            <select class="form-select"
                    name="technician_id"
                    id="technicianSelect"
                    required>

              <option value="">
                SELECT TECHNICIAN...
              </option>

              {% for technician in technicians %}

                <option value="{{ technician.id }}">
                  {{ technician.username|upper }}
                </option>

              {% endfor %}

            </select>

          </div>


          <div class="col-lg-6">

            <div class="issue-label">
              Work Order *
            </div>

            <div class="search-box">

              <input type="search"
                     class="form-control"
                     id="workOrderSearch"
                     placeholder="TYPE W/O #"
                     autocomplete="off">

              <input type="hidden"
                     name="work_order_id"
                     id="workOrderId">

              <div id="workOrderResults"
                   class="search-results d-none">
              </div>

            </div>

            <div id="selectedWorkOrder"
                 class="wo-selected d-none">

              <div class="fw-bold"
                   id="selectedWoNumber">
              </div>

              <div class="small text-muted"
                   id="selectedWoInfo">
              </div>

            </div>

          </div>

        </div>

      </div>
    </div>


    <!-- =====================================================
         AVAILABLE APPLIANCES
         ===================================================== -->

    <div class="issue-card mb-3">

      <div class="p-3 border-bottom">

        <div class="availability-header mb-2">

          <div>

            <div class="fw-bold">
              Available Appliances
            </div>

            <div class="small text-muted">
              Select category, then choose the physical appliance.
            </div>

          </div>

          <div class="text-end">

            <div class="small text-muted">
              AVAILABLE IN WAREHOUSE
            </div>

            <div class="fw-bold">
              <span id="totalAvailable">0</span>
              UNITS
            </div>

          </div>

        </div>


        <div id="categoryStrip"
             class="category-strip">

          <div class="text-muted small">
            Loading categories...
          </div>

        </div>

      </div>


      <!-- Filters -->

      <div class="p-3 border-bottom">

        <div class="row g-2">

          <div class="col-lg-3">

            <div class="issue-label">
              Brand
            </div>

            <select id="brandFilter"
                    class="form-select form-select-sm">

              <option value="">
                ALL BRANDS
              </option>

            </select>

          </div>


          <div class="col-lg-6">

            <div class="issue-label">
              Search
            </div>

            <input type="search"
                   id="inventorySearch"
                   class="form-control form-control-sm"
                   placeholder="MODEL / SERIAL / AP #">

          </div>


          <div class="col-lg-3 d-flex align-items-end">

            <button type="button"
                    class="btn btn-outline-secondary btn-sm"
                    id="clearInventoryFilters">
              Clear Filters
            </button>

          </div>

        </div>

      </div>


      <!-- Available units -->

      <div class="table-responsive">

        <table class="table table-sm table-hover available-table mb-0">

          <thead class="table-light">
            <tr>
              <th>APPLIANCE</th>
              <th>BRAND</th>
              <th>MODEL</th>
              <th>SERIAL</th>
              <th>SIZE</th>
              <th>CONDITION</th>
              <th>AP #</th>
              <th style="width:75px;"></th>
            </tr>
          </thead>

          <tbody id="availableBody">

            <tr>
              <td colspan="8"
                  class="loading-row">
                Loading available inventory...
              </td>
            </tr>

          </tbody>

        </table>

      </div>

      <div class="px-3 py-2 small text-muted border-top">
        Showing up to 100 matching AVAILABLE units.
        Use category, brand, model or serial to narrow the list.
      </div>

    </div>


    <!-- =====================================================
         SELECTED
         ===================================================== -->

    <div class="issue-card mb-3">

      <div class="p-3 border-bottom d-flex justify-content-between align-items-center">

        <div>

          <div class="fw-bold">
            Selected for Issue
          </div>

          <div class="small text-muted">
            These appliances will be placed on one Issue Slip.
          </div>

        </div>

        <div class="text-end">

          <div class="small text-muted">
            SELECTED
          </div>

          <div class="summary-count">
            <span id="selectedCount">0</span>
            APPLIANCES
          </div>

        </div>

      </div>


      <div class="table-responsive">

        <table class="table table-sm selected-table mb-0">

          <thead class="table-light">
            <tr>
              <th style="width:40px;">#</th>
              <th>APPLIANCE</th>
              <th>BRAND</th>
              <th>MODEL</th>
              <th>SERIAL</th>
              <th>SIZE</th>
              <th>CONDITION</th>
              <th>AP #</th>
              <th style="width:80px;"></th>
            </tr>
          </thead>

          <tbody id="selectedBody">

            <tr>
              <td colspan="9"
                  class="inventory-empty">
                No appliances selected.
              </td>
            </tr>

          </tbody>

        </table>

      </div>

    </div>


    <!-- NOTES -->

    <div class="issue-card mb-3">
      <div class="p-3">

        <div class="issue-label">
          Notes
        </div>

        <textarea class="form-control"
                  name="notes"
                  rows="2"
                  maxlength="1000"
                  placeholder="OPTIONAL"></textarea>

      </div>
    </div>


    <!-- ACTIONS -->

    <div class="d-flex justify-content-end gap-2 mb-4">

      <a class="btn btn-outline-secondary"
         href="{{ url_for('appliance.inventory_list') }}">
        Cancel
      </a>

      <button type="submit"
              class="btn btn-success btn-lg"
              id="issueButton"
              disabled>
        ISSUE & PRINT
      </button>

    </div>

  </form>

</div>


<script>
(() => {
  "use strict";


  // =========================================================
  // DOM
  // =========================================================

  const warehouseSelect =
    document.getElementById("warehouseSelect");

  const technicianSelect =
    document.getElementById("technicianSelect");

  const woSearch =
    document.getElementById("workOrderSearch");

  const woId =
    document.getElementById("workOrderId");

  const woResults =
    document.getElementById("workOrderResults");

  const selectedWoBox =
    document.getElementById("selectedWorkOrder");

  const selectedWoNumber =
    document.getElementById("selectedWoNumber");

  const selectedWoInfo =
    document.getElementById("selectedWoInfo");

  const categoryStrip =
    document.getElementById("categoryStrip");

  const brandFilter =
    document.getElementById("brandFilter");

  const inventorySearch =
    document.getElementById("inventorySearch");

  const clearInventoryFilters =
    document.getElementById("clearInventoryFilters");

  const availableBody =
    document.getElementById("availableBody");

  const totalAvailable =
    document.getElementById("totalAvailable");

  const selectedBody =
    document.getElementById("selectedBody");

  const selectedCount =
    document.getElementById("selectedCount");

  const issueButton =
    document.getElementById("issueButton");

  const issueForm =
    document.getElementById("issueForm");


  // =========================================================
  // STATE
  // =========================================================

  const selectedUnits = new Map();

  let selectedCategoryId = 0;

  let inventoryTimer = null;
  let woTimer = null;

  let lastCatalogItems = [];


  const applianceSearchUrl =
    {{ url_for(
      'appliance.issue_search_appliances'
    )|tojson }};

  const workOrderSearchUrl =
    {{ url_for(
      'appliance.issue_search_work_orders'
    )|tojson }};


  // =========================================================
  // HELPERS
  // =========================================================

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }


  function hideWoResults() {
    woResults.classList.add("d-none");
    woResults.innerHTML = "";
  }


  function updateSubmitState() {

    issueButton.disabled = !(
      warehouseSelect.value
      && technicianSelect.value
      && woId.value
      && selectedUnits.size > 0
    );
  }


  // =========================================================
  // WORK ORDER
  // =========================================================

  function clearWorkOrder() {

    woId.value = "";

    selectedWoBox.classList.add(
      "d-none"
    );

    selectedWoNumber.textContent = "";
    selectedWoInfo.textContent = "";

    updateSubmitState();
  }


  function selectWorkOrder(item) {

    woId.value = item.id;

    const number =
      item.job_numbers
      || item.canonical_job
      || "";

    woSearch.value = number;

    selectedWoNumber.textContent =
      "W/O: " + number;

    const info = [];

    if (item.technician) {
      info.push(
        "ASSIGNED TECH: "
        + item.technician
      );
    }

    if (item.customer_po) {
      info.push(
        "PO: "
        + item.customer_po
      );
    }

    selectedWoInfo.textContent =
      info.join(" | ");

    selectedWoBox.classList.remove(
      "d-none"
    );

    hideWoResults();

    updateSubmitState();
  }


  async function searchWorkOrders() {

    const q =
      woSearch.value.trim();

    if (q.length < 2) {
      hideWoResults();
      return;
    }

    try {

      const response =
        await fetch(
          workOrderSearchUrl
          + "?q="
          + encodeURIComponent(q)
        );

      const data =
        await response.json();

      if (!response.ok || !data.ok) {
        hideWoResults();
        return;
      }

      if (!data.items.length) {

        woResults.innerHTML = `
          <div class="search-result text-muted">
            NO WORK ORDERS FOUND
          </div>
        `;

        woResults.classList.remove(
          "d-none"
        );

        return;
      }

      woResults.innerHTML =
        data.items.map(
          item => `
            <div class="search-result"
                 data-wo="${item.id}">

              <div class="fw-bold">
                ${esc(item.job_numbers)}
              </div>

              <div class="small text-muted">

                ${
                  item.technician
                  ? "TECH: "
                    + esc(item.technician)
                  : ""
                }

                ${
                  item.customer_po
                  ? " | PO: "
                    + esc(item.customer_po)
                  : ""
                }

              </div>

            </div>
          `
        ).join("");

      woResults.classList.remove(
        "d-none"
      );

      woResults
        .querySelectorAll("[data-wo]")
        .forEach(
          element => {

            element.addEventListener(
              "click",
              () => {

                const id =
                  Number(
                    element.dataset.wo
                  );

                const item =
                  data.items.find(
                    row => row.id === id
                  );

                if (item) {
                  selectWorkOrder(item);
                }
              }
            );
          }
        );

    } catch (error) {
      console.error(error);
      hideWoResults();
    }
  }


  // =========================================================
  // CATALOG
  // =========================================================

  function renderCategories(
    categories,
    total
  ) {

    let html = `
      <button type="button"
              class="category-btn ${
                selectedCategoryId === 0
                ? "active"
                : ""
              }"
              data-category-id="0">

        <div class="category-name">
          ALL
        </div>

        <div class="category-count">
          ${total} AVAILABLE
        </div>

      </button>
    `;

    html += categories.map(
      category => `
        <button type="button"
                class="category-btn ${
                  Number(category.id)
                    === selectedCategoryId
                  ? "active"
                  : ""
                }"
                data-category-id="${category.id}">

          <div class="category-name">
            ${esc(category.name)}
          </div>

          <div class="category-count">
            ${category.count} AVAILABLE
          </div>

        </button>
      `
    ).join("");

    categoryStrip.innerHTML = html;

    categoryStrip
      .querySelectorAll(
        "[data-category-id]"
      )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              selectedCategoryId =
                Number(
                  button.dataset.categoryId
                );

              brandFilter.value = "";
              inventorySearch.value = "";

              loadCatalog();
            }
          );
        }
      );
  }


  function renderBrands(brands) {

    const current =
      brandFilter.value;

    brandFilter.innerHTML = `
      <option value="">
        ALL BRANDS
      </option>
    `;

    for (const brand of brands) {

      const option =
        document.createElement(
          "option"
        );

      option.value = brand;
      option.textContent = brand;

      brandFilter.appendChild(option);
    }

    if (
      current
      && brands.includes(current)
    ) {
      brandFilter.value = current;
    }
  }


  function renderAvailable(items) {

    lastCatalogItems = items;

    const filtered =
      items.filter(
        item => !selectedUnits.has(
          Number(item.id)
        )
      );

    if (!filtered.length) {

      availableBody.innerHTML = `
        <tr>
          <td colspan="8"
              class="inventory-empty">
            No matching AVAILABLE appliances.
          </td>
        </tr>
      `;

      return;
    }

    availableBody.innerHTML =
      filtered.map(
        item => `
          <tr>

            <td class="fw-bold">
              ${esc(item.appliance_type)}
            </td>

            <td>
              ${esc(item.brand || "—")}
            </td>

            <td>
              ${esc(item.model || "—")}
            </td>

            <td class="fw-semibold">
              ${esc(item.serial || "—")}
            </td>

            <td>
              ${esc(item.size || "—")}
            </td>

            <td>
              ${esc(item.condition || "—")}
            </td>

            <td>
              <span class="ap-number">
                ${esc(item.inventory_number)}
              </span>
            </td>

            <td>
              <button type="button"
                      class="btn btn-primary btn-sm"
                      data-add="${item.id}">
                ADD
              </button>
            </td>

          </tr>
        `
      ).join("");

    availableBody
      .querySelectorAll(
        "[data-add]"
      )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              const id =
                Number(
                  button.dataset.add
                );

              const item =
                items.find(
                  row =>
                    Number(row.id)
                    === id
                );

              if (item) {
                addUnit(item);
              }
            }
          );
        }
      );
  }


  async function loadCatalog() {

    const warehouseId =
      warehouseSelect.value;

    if (!warehouseId) {
      return;
    }

    availableBody.innerHTML = `
      <tr>
        <td colspan="8"
            class="loading-row">
          Loading...
        </td>
      </tr>
    `;

    const params =
      new URLSearchParams();

    params.set(
      "warehouse_id",
      warehouseId
    );

    if (selectedCategoryId > 0) {
      params.set(
        "category_id",
        String(selectedCategoryId)
      );
    }

    if (brandFilter.value) {
      params.set(
        "brand",
        brandFilter.value
      );
    }

    if (inventorySearch.value.trim()) {
      params.set(
        "q",
        inventorySearch.value.trim()
      );
    }

    try {

      const response =
        await fetch(
          applianceSearchUrl
          + "?"
          + params.toString()
        );

      const data =
        await response.json();

      if (!response.ok || !data.ok) {

        availableBody.innerHTML = `
          <tr>
            <td colspan="8"
                class="inventory-empty text-danger">
              Unable to load inventory.
            </td>
          </tr>
        `;

        return;
      }

      totalAvailable.textContent =
        String(
          data.total_available || 0
        );

      renderCategories(
        data.categories || [],
        data.total_available || 0
      );

      renderBrands(
        data.brands || []
      );

      renderAvailable(
        data.items || []
      );

    } catch (error) {

      console.error(error);

      availableBody.innerHTML = `
        <tr>
          <td colspan="8"
              class="inventory-empty text-danger">
            Unable to load inventory.
          </td>
        </tr>
      `;
    }
  }


  // =========================================================
  // SELECTED
  // =========================================================

  function addUnit(item) {

    selectedUnits.set(
      Number(item.id),
      item
    );

    renderSelected();
    renderAvailable(
      lastCatalogItems
    );
  }


  function removeUnit(id) {

    selectedUnits.delete(
      Number(id)
    );

    renderSelected();

    loadCatalog();
  }


  function renderSelected() {

    selectedCount.textContent =
      String(
        selectedUnits.size
      );

    if (!selectedUnits.size) {

      selectedBody.innerHTML = `
        <tr>
          <td colspan="9"
              class="inventory-empty">
            No appliances selected.
          </td>
        </tr>
      `;

      updateSubmitState();
      return;
    }

    let index = 1;

    let html = "";

    for (
      const item
      of selectedUnits.values()
    ) {

      html += `
        <tr class="selected-row">

          <td>
            ${index}
          </td>

          <td class="fw-bold">
            ${esc(item.appliance_type)}
          </td>

          <td>
            ${esc(item.brand || "—")}
          </td>

          <td>
            ${esc(item.model || "—")}
          </td>

          <td class="fw-semibold">
            ${esc(item.serial || "—")}
          </td>

          <td>
            ${esc(item.size || "—")}
          </td>

          <td>
            ${esc(item.condition || "—")}
          </td>

          <td>
            <span class="ap-number">
              ${esc(item.inventory_number)}
            </span>

            <input type="hidden"
                   name="appliance_unit_ids"
                   value="${item.id}">
          </td>

          <td>
            <button type="button"
                    class="btn btn-outline-danger btn-sm"
                    data-remove="${item.id}">
              REMOVE
            </button>
          </td>

        </tr>
      `;

      index++;
    }

    selectedBody.innerHTML = html;

    selectedBody
      .querySelectorAll(
        "[data-remove]"
      )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              removeUnit(
                Number(
                  button.dataset.remove
                )
              );
            }
          );
        }
      );

    updateSubmitState();
  }


  // =========================================================
  // EVENTS
  // =========================================================

  woSearch.addEventListener(
    "input",
    () => {

      clearWorkOrder();

      clearTimeout(
        woTimer
      );

      woTimer =
        setTimeout(
          searchWorkOrders,
          220
        );
    }
  );


  warehouseSelect.addEventListener(
    "change",
    () => {

      if (selectedUnits.size) {

        const ok = confirm(
          "Changing warehouse will clear "
          + "selected appliances. Continue?"
        );

        if (!ok) {
          return;
        }
      }

      selectedUnits.clear();

      selectedCategoryId = 0;

      brandFilter.value = "";
      inventorySearch.value = "";

      renderSelected();

      loadCatalog();
    }
  );


  brandFilter.addEventListener(
    "change",
    loadCatalog
  );


  inventorySearch.addEventListener(
    "input",
    () => {

      clearTimeout(
        inventoryTimer
      );

      inventoryTimer =
        setTimeout(
          loadCatalog,
          250
        );
    }
  );


  clearInventoryFilters.addEventListener(
    "click",
    () => {

      selectedCategoryId = 0;

      brandFilter.value = "";
      inventorySearch.value = "";

      loadCatalog();
    }
  );


  technicianSelect.addEventListener(
    "change",
    updateSubmitState
  );


  document.addEventListener(
    "click",
    event => {

      if (
        !event.target.closest(
          "#workOrderSearch, #workOrderResults"
        )
      ) {
        hideWoResults();
      }
    }
  );


  document.addEventListener(
    "input",
    event => {

      const element =
        event.target;

      if (
        element.matches(
          '.issue-page input[type="text"],'
          + '.issue-page input[type="search"],'
          + '.issue-page textarea'
        )
      ) {

        const start =
          element.selectionStart;

        const end =
          element.selectionEnd;

        element.value =
          element.value.toUpperCase();

        if (
          start !== null
          && end !== null
          && typeof element.setSelectionRange
             === "function"
        ) {

          element.setSelectionRange(
            start,
            end
          );
        }
      }
    }
  );


  issueForm.addEventListener(
    "submit",
    event => {

      if (!technicianSelect.value) {

        event.preventDefault();

        alert(
          "SELECT TECHNICIAN."
        );

        technicianSelect.focus();

        return;
      }

      if (!woId.value) {

        event.preventDefault();

        alert(
          "SELECT WORK ORDER."
        );

        woSearch.focus();

        return;
      }

      if (!selectedUnits.size) {

        event.preventDefault();

        alert(
          "SELECT AT LEAST ONE APPLIANCE."
        );

        return;
      }

      issueButton.disabled = true;

      issueButton.textContent =
        "ISSUING...";
    }
  );


  // =========================================================
  // INITIAL LOAD
  // =========================================================

  renderSelected();
  loadCatalog();

})();
</script>

{% endblock %}
'''


template_path.write_text(
    new_template,
    encoding="utf-8",
)


print("=" * 70)
print("APPLIANCE ISSUE CATALOG UI")
print("=" * 70)
print("OK: appliance/routes.py updated")
print("OK: templates/appliance_issue_new.html replaced")
print()
print("Backups:")
print(" ", routes_backup)
print(" ", template_backup)
print("=" * 70)
