from pathlib import Path
from datetime import datetime


STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

routes_path = Path("appliance/routes.py")
template_path = Path(
    "templates/appliance_receiving_form.html"
)


def backup(path: Path):
    dst = Path(
        str(path)
        + f".before_serial_scan_check_{STAMP}.bak"
    )

    dst.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print("BACKUP:", dst)


for path in [
    routes_path,
    template_path,
]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found"
        )

    backup(path)


# ============================================================
# 1. BACKEND SERIAL CHECK
# ============================================================

routes_text = routes_path.read_text(
    encoding="utf-8"
)


if "def receiving_check_serial(" not in routes_text:

    marker = '''# ============================================================
# Bulk Add physical appliances
# ============================================================
'''

    if marker not in routes_text:
        raise SystemExit(
            "ERROR: Bulk Add marker not found"
        )


    endpoint = r'''# ============================================================
# Receiving Serial duplicate check
#
# Used by barcode scanner / Batch Entry before moving to
# the next Serial row.
# ============================================================

@appliance_bp.get(
    "/receiving/check-serial"
)
@login_required
def receiving_check_serial():

    serial = (
        request.args.get("serial")
        or ""
    ).strip().upper()

    try:
        receiving_id = int(
            request.args.get("receiving_id")
            or 0
        )
    except (TypeError, ValueError):
        receiving_id = 0

    if not serial:

        return jsonify(
            {
                "ok": True,
                "exists": False,
            }
        )

    # --------------------------------------------------------
    # Verify current Receiving + warehouse access
    # --------------------------------------------------------

    receiving = db.session.get(
        ApplianceReceiving,
        receiving_id,
    )

    if receiving is None:

        return jsonify(
            {
                "ok": False,
                "error": "Receiving not found.",
            }
        ), 404

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=receiving.warehouse_id,
    ):

        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # 1. Already exists as physical ApplianceUnit
    # --------------------------------------------------------

    existing_unit = (
        ApplianceUnit.query
        .filter(
            db.func.upper(
                db.func.trim(
                    ApplianceUnit.serial_number
                )
            )
            == serial
        )
        .first()
    )

    if existing_unit is not None:

        return jsonify(
            {
                "ok": True,
                "exists": True,
                "source": "inventory",
                "message": (
                    f"SERIAL {serial} already exists "
                    f"in Inventory as "
                    f"{existing_unit.inventory_number}."
                ),
                "inventory_number":
                    existing_unit.inventory_number,
            }
        )

    # --------------------------------------------------------
    # 2. Already exists in a saved Receiving line
    #
    # This also catches a Serial entered in another Draft
    # Receiving before it becomes an ApplianceUnit.
    # --------------------------------------------------------

    existing_line = (
        ApplianceReceivingLine.query
        .filter(
            db.func.upper(
                db.func.trim(
                    ApplianceReceivingLine.serial_number
                )
            )
            == serial
        )
        .first()
    )

    if existing_line is not None:

        source_receiving = db.session.get(
            ApplianceReceiving,
            existing_line.receiving_id,
        )

        receiving_number = (
            source_receiving.receiving_number
            if source_receiving is not None
            else ""
        )

        receiving_status = (
            source_receiving.status
            if source_receiving is not None
            else ""
        )

        return jsonify(
            {
                "ok": True,
                "exists": True,
                "source": "receiving",
                "message": (
                    f"SERIAL {serial} already exists "
                    f"in Receiving "
                    f"{receiving_number or '#'+str(existing_line.receiving_id)}"
                    f"{' (' + receiving_status.upper() + ')' if receiving_status else ''}."
                ),
                "receiving_id":
                    existing_line.receiving_id,
                "receiving_number":
                    receiving_number,
            }
        )

    return jsonify(
        {
            "ok": True,
            "exists": False,
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
        "OK: receiving_check_serial endpoint added"
    )

else:

    print(
        "SKIP: receiving_check_serial already exists"
    )


# ============================================================
# 2. TEMPLATE
# ============================================================

template_text = template_path.read_text(
    encoding="utf-8"
)


# ------------------------------------------------------------
# Add URL constants
# ------------------------------------------------------------

if "const serialCheckUrl" not in template_text:

    needle = '''  const canPricing = {{ 'true' if can_pricing else 'false' }};
'''

    replacement = '''  const canPricing = {{ 'true' if can_pricing else 'false' }};

  const serialCheckUrl = {{ url_for(
    'appliance.receiving_check_serial'
  )|tojson }};

  const currentReceivingId = {{ receiving.id }};
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
        "OK: Serial check URL added"
    )


# ------------------------------------------------------------
# Add Serial validation functions before saveAll()
# ------------------------------------------------------------

if "async function validateScannedSerial(" not in template_text:

    marker = '''  async function saveAll() {
'''

    if marker not in template_text:
        raise SystemExit(
            "ERROR: saveAll() not found"
        )

    functions = r'''  function normalizeSerial(value) {
    return String(
      value
      || ""
    )
      .trim()
      .toUpperCase();
  }


  function clearSerialError(input) {

    input.classList.remove(
      "is-invalid"
    );

    input.removeAttribute(
      "title"
    );
  }


  function setSerialError(
    input,
    message
  ) {

    input.classList.add(
      "is-invalid"
    );

    input.setAttribute(
      "title",
      message
    );

    showError(
      message
    );

    input.focus();

    input.select();
  }


  function duplicateSerialInBatch(
    input,
    serial
  ) {

    const normalized =
      normalizeSerial(serial);

    if (!normalized) {
      return null;
    }

    const serialInputs = [
      ...body.querySelectorAll(
        '[data-field="serial_number"]'
      )
    ];

    for (
      const other
      of serialInputs
    ) {

      if (other === input) {
        continue;
      }

      if (
        normalizeSerial(
          other.value
        ) === normalized
      ) {
        return other;
      }
    }

    return null;
  }


  async function validateScannedSerial(
    input
  ) {

    clearMessages();
    clearSerialError(input);

    const serial =
      normalizeSerial(
        input.value
      );

    input.value =
      serial;

    if (!serial) {

      setSerialError(
        input,
        "SERIAL IS REQUIRED BEFORE MOVING TO THE NEXT ROW."
      );

      return false;
    }

    # --------------------------------------------------------
    # NOTE:
    # '#' cannot be used for JS comments.
    # This text is replaced below with // automatically.
    # --------------------------------------------------------

    const duplicateInput =
      duplicateSerialInBatch(
        input,
        serial
      );

    if (duplicateInput) {

      const rows = [
        ...body.querySelectorAll("tr")
      ];

      const duplicateRow =
        rows.indexOf(
          duplicateInput.closest("tr")
        ) + 1;

      setSerialError(
        input,
        `SERIAL ${serial} IS ALREADY ENTERED IN BATCH ROW ${duplicateRow}.`
      );

      return false;
    }

    try {

      const params =
        new URLSearchParams();

      params.set(
        "serial",
        serial
      );

      params.set(
        "receiving_id",
        String(
          currentReceivingId
        )
      );

      const response =
        await fetch(
          `${serialCheckUrl}?${params.toString()}`,
          {
            headers: {
              "Accept":
                "application/json"
            }
          }
        );

      const result =
        await response.json();

      if (
        !response.ok
        || !result.ok
      ) {

        setSerialError(
          input,
          result.error
          || "UNABLE TO VERIFY SERIAL."
        );

        return false;
      }

      if (result.exists) {

        setSerialError(
          input,
          result.message
          || `SERIAL ${serial} ALREADY EXISTS.`
        );

        return false;
      }

      clearSerialError(
        input
      );

      return true;

    } catch (error) {

      console.error(
        "Serial validation error:",
        error
      );

      setSerialError(
        input,
        "UNABLE TO VERIFY SERIAL. CHECK CONNECTION AND TRY AGAIN."
      );

      return false;
    }
  }


'''

    # Python patch cleanup:
    # convert explanatory accidental # JS lines into //
    functions = functions.replace(
        "    # --------------------------------------------------------",
        "    // --------------------------------------------------------"
    ).replace(
        "    # NOTE:",
        "    // NOTE:"
    ).replace(
        "    # '#' cannot be used for JS comments.",
        "    // '#' cannot be used for JS comments."
    ).replace(
        "    # This text is replaced below with // automatically.",
        "    // Serial is checked locally before server validation."
    )

    template_text = template_text.replace(
        marker,
        functions + marker,
        1,
    )

    print(
        "OK: Serial validation functions added"
    )


# ------------------------------------------------------------
# Replace old ENTER-on-SERIAL handler
# ------------------------------------------------------------

old_handler = '''  body.addEventListener(
    "keydown",
    event => {
      const target = event.target;

      if (
        event.key !== "Enter"
        || target?.dataset?.field !== "serial_number"
      ) {
        return;
      }

      event.preventDefault();

      const currentRow =
        target.closest("tr");

      const data =
        rowData(currentRow);

      data.serial_number = "";

      addRow(
        data,
        {
          focusSerial:true
        }
      );
    }
  );
'''

new_handler = '''  body.addEventListener(
    "keydown",
    async event => {

      const target =
        event.target;

      if (
        event.key !== "Enter"
        || target?.dataset?.field
           !== "serial_number"
      ) {
        return;
      }

      /*
        Barcode scanners normally send:

            SERIAL VALUE
            ENTER

        Stop normal form behavior and validate the
        scanned Serial before creating the next row.
      */

      event.preventDefault();
      event.stopPropagation();

      if (
        target.dataset.serialChecking
        === "1"
      ) {
        return;
      }

      target.dataset.serialChecking =
        "1";

      try {

        const valid =
          await validateScannedSerial(
            target
          );

        if (!valid) {
          return;
        }

        const currentRow =
          target.closest("tr");

        if (!currentRow) {
          return;
        }

        const data =
          rowData(
            currentRow
          );

        /*
          Preserve all current appliance information.
          Only Serial is cleared for the next physical unit.
        */

        data.serial_number = "";

        addRow(
          data,
          {
            focusSerial:true
          }
        );

        showStatus(
          "SERIAL ACCEPTED — READY FOR NEXT SCAN."
        );

      } finally {

        target.dataset.serialChecking =
          "0";
      }
    }
  );
'''


if old_handler in template_text:

    template_text = template_text.replace(
        old_handler,
        new_handler,
        1,
    )

    print(
        "OK: Serial ENTER/Scanner handler upgraded"
    )

elif "SERIAL ACCEPTED — READY FOR NEXT SCAN." in template_text:

    print(
        "SKIP: Scanner ENTER handler already upgraded"
    )

else:

    raise SystemExit(
        "ERROR: original Serial keydown handler not found"
    )


# ------------------------------------------------------------
# Clear red error automatically when user changes Serial
# ------------------------------------------------------------

if "clearSerialError(event.target);" not in template_text:

    marker = '''  body.addEventListener(
    "keydown",
    async event => {
'''

    if marker not in template_text:
        raise SystemExit(
            "ERROR: upgraded keydown marker not found"
        )

    input_handler = '''  body.addEventListener(
    "input",
    event => {

      if (
        event.target?.dataset?.field
        === "serial_number"
      ) {

        clearSerialError(
          event.target
        );
      }
    }
  );


'''

    template_text = template_text.replace(
        marker,
        input_handler + marker,
        1,
    )

    print(
        "OK: Serial error reset added"
    )


template_path.write_text(
    template_text,
    encoding="utf-8",
)


print()
print("=" * 70)
print("APPLIANCE RECEIVING SCANNER PATCH COMPLETE")
print("=" * 70)
print("SCAN -> ENTER -> DUPLICATE CHECK -> NEXT SERIAL")
print()
print("Checks:")
print("  1. Duplicate inside current Batch Entry")
print("  2. Duplicate in saved Receiving lines")
print("  3. Duplicate in physical Appliance Inventory")
print()
print("On success:")
print("  Appliance / Brand / Model remain")
print("  Size / Unit / Condition remain")
print("  Serial becomes empty")
print("  Cursor moves directly to next Serial")
print("=" * 70)

