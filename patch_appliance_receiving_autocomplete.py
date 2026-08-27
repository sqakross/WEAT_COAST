from pathlib import Path
from datetime import datetime


STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

routes_path = Path("appliance/routes.py")
template_path = Path("templates/appliance_receiving_form.html")


def backup(path: Path):
    dst = Path(
        str(path)
        + f".before_receiving_autocomplete_{STAMP}.bak"
    )

    dst.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print("BACKUP:", dst)


for path in [routes_path, template_path]:

    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found"
        )

    backup(path)


# ============================================================
# 1. BACKEND ENDPOINT
# ============================================================

routes_text = routes_path.read_text(
    encoding="utf-8"
)

if "def receiving_autocomplete(" not in routes_text:

    marker = '''# ============================================================
# Bulk Add physical appliances
# ============================================================
'''

    if marker not in routes_text:
        raise SystemExit(
            "ERROR: Bulk Add marker not found in appliance/routes.py"
        )

    endpoint = r'''# ============================================================
# Receiving autocomplete
#
# Historical Brand / Model suggestions from ApplianceUnit.
#
# IMPORTANT:
# - not limited to AVAILABLE inventory;
# - new Brand / Model values are still allowed;
# - suggestions respect category;
# - model suggestions respect category + brand.
# ============================================================

@appliance_bp.get(
    "/receiving/autocomplete"
)
@login_required
def receiving_autocomplete():

    from sqlalchemy import func

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

    if category_id <= 0:
        return jsonify(
            {
                "ok": True,
                "brands": [],
                "models": [],
            }
        )

    # --------------------------------------------------------
    # Access
    # --------------------------------------------------------

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.receive"
    )

    if not allowed_warehouse_ids:
        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # BRAND suggestions
    #
    # We intentionally use the complete ApplianceUnit registry,
    # including issued/sold/etc. units.
    # This behaves as historical warehouse catalog.
    # --------------------------------------------------------

    brand_rows = (
        db.session.query(
            ApplianceUnit.brand
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceUnit.category_id
            == category_id,
            ApplianceUnit.brand.isnot(None),
            func.trim(
                ApplianceUnit.brand
            ) != "",
        )
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
    # MODEL suggestions
    # --------------------------------------------------------

    models = []

    if brand:

        model_rows = (
            db.session.query(
                ApplianceUnit.model_number
            )
            .filter(
                ApplianceUnit.warehouse_id.in_(
                    allowed_warehouse_ids
                ),
                ApplianceUnit.category_id
                == category_id,
                func.upper(
                    func.trim(
                        ApplianceUnit.brand
                    )
                )
                == brand,
                ApplianceUnit.model_number.isnot(
                    None
                ),
                func.trim(
                    ApplianceUnit.model_number
                ) != "",
            )
            .distinct()
            .order_by(
                ApplianceUnit.model_number.asc()
            )
            .all()
        )

        models = sorted(
            {
                (
                    row[0]
                    or ""
                ).strip().upper()
                for row in model_rows
                if (
                    row[0]
                    or ""
                ).strip()
            }
        )

    return jsonify(
        {
            "ok": True,
            "brands": brands,
            "models": models,
        }
    )


'''

    routes_text = routes_text.replace(
        marker,
        endpoint + marker,
        1,
    )

    routes_path.write_text(
        routes_text,
        encoding="utf-8",
    )

    print(
        "OK: receiving_autocomplete endpoint added"
    )

else:

    print(
        "SKIP: receiving_autocomplete already exists"
    )


# ============================================================
# 2. TEMPLATE
# ============================================================

template_text = template_path.read_text(
    encoding="utf-8"
)


# ------------------------------------------------------------
# Autocomplete URL + cache
# ------------------------------------------------------------

if "const autocompleteUrl" not in template_text:

    needle = '''  const canPricing = {{ 'true' if can_pricing else 'false' }};
'''

    replacement = '''  const canPricing = {{ 'true' if can_pricing else 'false' }};

  const autocompleteUrl = {{ url_for(
    'appliance.receiving_autocomplete'
  )|tojson }};

  const autocompleteCache = new Map();
'''

    if needle not in template_text:
        raise SystemExit(
            "ERROR: canPricing line not found"
        )

    template_text = template_text.replace(
        needle,
        replacement,
        1,
    )

    print(
        "OK: autocomplete URL added"
    )


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

if "async function loadAutocomplete(" not in template_text:

    marker = '''  function addRow(data = {}, options = {}) {
'''

    if marker not in template_text:
        raise SystemExit(
            "ERROR: addRow function not found"
        )

    functions = r'''  async function loadAutocomplete(
    categoryId,
    brand = ""
  ) {

    const category =
      String(
        categoryId
        || ""
      ).trim();

    const brandValue =
      String(
        brand
        || ""
      )
      .trim()
      .toUpperCase();

    if (!category) {

      return {
        brands: [],
        models: []
      };
    }

    const cacheKey =
      `${category}|${brandValue}`;

    if (
      autocompleteCache.has(
        cacheKey
      )
    ) {
      return autocompleteCache.get(
        cacheKey
      );
    }

    const params =
      new URLSearchParams();

    params.set(
      "category_id",
      category
    );

    if (brandValue) {

      params.set(
        "brand",
        brandValue
      );
    }

    try {

      const response =
        await fetch(
          `${autocompleteUrl}?${params.toString()}`,
          {
            headers: {
              "Accept":
                "application/json"
            }
          }
        );

      if (!response.ok) {

        return {
          brands: [],
          models: []
        };
      }

      const result =
        await response.json();

      const data = {

        brands:
          Array.isArray(
            result.brands
          )
            ? result.brands
            : [],

        models:
          Array.isArray(
            result.models
          )
            ? result.models
            : []
      };

      autocompleteCache.set(
        cacheKey,
        data
      );

      return data;

    } catch (error) {

      console.error(
        "Autocomplete error:",
        error
      );

      return {
        brands: [],
        models: []
      };
    }
  }


  function ensureDatalist(
    input,
    suffix
  ) {

    let listId =
      input.dataset.autocompleteList;

    if (!listId) {

      listId =
        `appliance-${suffix}-`
        + Math.random()
          .toString(36)
          .slice(2);

      input.dataset.autocompleteList =
        listId;

      input.setAttribute(
        "list",
        listId
      );

      const datalist =
        document.createElement(
          "datalist"
        );

      datalist.id =
        listId;

      document.body.appendChild(
        datalist
      );
    }

    return document.getElementById(
      listId
    );
  }


  function fillDatalist(
    datalist,
    values
  ) {

    datalist.innerHTML = "";

    for (
      const value
      of values
    ) {

      const option =
        document.createElement(
          "option"
        );

      option.value =
        value;

      datalist.appendChild(
        option
      );
    }
  }


  async function refreshBrandSuggestions(
    row
  ) {

    const categoryInput =
      row.querySelector(
        '[data-field="category_id"]'
      );

    const brandInput =
      row.querySelector(
        '[data-field="brand"]'
      );

    if (
      !categoryInput
      || !brandInput
    ) {
      return;
    }

    const datalist =
      ensureDatalist(
        brandInput,
        "brand"
      );

    if (!categoryInput.value) {

      fillDatalist(
        datalist,
        []
      );

      return;
    }

    const result =
      await loadAutocomplete(
        categoryInput.value
      );

    fillDatalist(
      datalist,
      result.brands
    );
  }


  async function refreshModelSuggestions(
    row
  ) {

    const categoryInput =
      row.querySelector(
        '[data-field="category_id"]'
      );

    const brandInput =
      row.querySelector(
        '[data-field="brand"]'
      );

    const modelInput =
      row.querySelector(
        '[data-field="model_number"]'
      );

    if (
      !categoryInput
      || !brandInput
      || !modelInput
    ) {
      return;
    }

    const datalist =
      ensureDatalist(
        modelInput,
        "model"
      );

    if (
      !categoryInput.value
      || !brandInput.value.trim()
    ) {

      fillDatalist(
        datalist,
        []
      );

      return;
    }

    const result =
      await loadAutocomplete(
        categoryInput.value,
        brandInput.value
      );

    fillDatalist(
      datalist,
      result.models
    );
  }


'''

    template_text = template_text.replace(
        marker,
        functions + marker,
        1,
    )

    print(
        "OK: autocomplete functions added"
    )


# ------------------------------------------------------------
# Connect autocomplete to every dynamically-created Batch row
# ------------------------------------------------------------

if "refreshBrandSuggestions(row);" not in template_text:

    needle = '''    const condition = row.querySelector(
      '[data-field="condition"]'
    );
'''

    replacement = '''    const categoryInput = row.querySelector(
      '[data-field="category_id"]'
    );

    const brandInput = row.querySelector(
      '[data-field="brand"]'
    );

    const modelInput = row.querySelector(
      '[data-field="model_number"]'
    );


    categoryInput.addEventListener(
      "change",
      async () => {

        /*
          Category changed:
          keep free-text Brand / Model,
          but refresh suggestion dictionaries.
        */

        await refreshBrandSuggestions(
          row
        );

        await refreshModelSuggestions(
          row
        );
      }
    );


    brandInput.addEventListener(
      "focus",
      () => {

        refreshBrandSuggestions(
          row
        );
      }
    );


    brandInput.addEventListener(
      "input",
      () => {

        brandInput.value =
          brandInput.value.toUpperCase();
      }
    );


    brandInput.addEventListener(
      "change",
      () => {

        brandInput.value =
          brandInput.value
            .trim()
            .toUpperCase();

        refreshModelSuggestions(
          row
        );
      }
    );


    modelInput.addEventListener(
      "focus",
      () => {

        refreshModelSuggestions(
          row
        );
      }
    );


    modelInput.addEventListener(
      "input",
      () => {

        modelInput.value =
          modelInput.value.toUpperCase();
      }
    );


    /*
      If duplicated/pasted row already has category / brand,
      prepare suggestions immediately.
    */

    if (categoryInput.value) {

      refreshBrandSuggestions(
        row
      );

      if (brandInput.value.trim()) {

        refreshModelSuggestions(
          row
        );
      }
    }


    const condition = row.querySelector(
      '[data-field="condition"]'
    );
'''

    if needle not in template_text:
        raise SystemExit(
            "ERROR: condition block inside addRow not found"
        )

    template_text = template_text.replace(
        needle,
        replacement,
        1,
    )

    print(
        "OK: Batch Entry events added"
    )

else:

    print(
        "SKIP: Batch autocomplete events already exist"
    )


template_path.write_text(
    template_text,
    encoding="utf-8",
)


print()
print("=" * 70)
print("RECEIVING BRAND / MODEL AUTOCOMPLETE PATCH COMPLETE")
print("=" * 70)
print("Brand: filtered by Appliance category")
print("Model: filtered by Appliance category + Brand")
print("New values remain allowed")
print("Serial remains manual / scanner input")
print("Duplicate / Excel Paste / Save All preserved")
print("=" * 70)

