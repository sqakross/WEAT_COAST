from pathlib import Path

path = Path("templates/appliance_issue_new.html")

text = path.read_text(
    encoding="utf-8"
)

# ------------------------------------------------------------
# Replace W/O HTML
# ------------------------------------------------------------

start = text.index(
    '          <div class="col-lg-6">\n\n            <div class="issue-label">\n              Work Order *'
)

end = text.index(
    '\n          </div>\n\n        </div>',
    start
) + len('\n          </div>')

new_block = '''          <div class="col-lg-6">

            <div class="issue-label">
              Work Order # *
            </div>

            <input type="text"
                   class="form-control"
                   name="work_order_number"
                   id="workOrderNumber"
                   maxlength="120"
                   placeholder="ENTER W/O #"
                   required>

            <div class="small text-muted mt-1">
              Enter the warehouse/customer work order number manually.
            </div>

          </div>'''

text = (
    text[:start]
    + new_block
    + text[end:]
)

# ------------------------------------------------------------
# JS: remove old W/O search references
# ------------------------------------------------------------

replacements = {
'''  const woSearch =
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
''':
'''  const workOrderNumber =
    document.getElementById("workOrderNumber");
''',

'''  let woTimer = null;
''':
'',

'''  const workOrderSearchUrl =
    {{ url_for(
      'appliance.issue_search_work_orders'
    )|tojson }};
''':
'',

'''      && woId.value
''':
'''      && workOrderNumber.value.trim()
''',
}

for old, new in replacements.items():
    if old not in text:
        raise SystemExit(
            "ERROR: expected JS block not found"
        )

    text = text.replace(
        old,
        new,
        1,
    )


# Remove old Work Order helper section.
wo_start = text.index(
    '  // =========================================================\n  // WORK ORDER\n  // ========================================================='
)

wo_end = text.index(
    '  // =========================================================\n  // CATALOG\n  // =========================================================',
    wo_start
)

text = (
    text[:wo_start]
    + text[wo_end:]
)


# Remove old woSearch event.
event_start = text.find(
    '''  woSearch.addEventListener(
    "input",
'''
)

if event_start >= 0:

    event_end = text.index(
        '''  warehouseSelect.addEventListener(
''',
        event_start,
    )

    text = (
        text[:event_start]
        + text[event_end:]
    )


# Replace technician event with technician + WO.
old = '''  technicianSelect.addEventListener(
    "change",
    updateSubmitState
  );
'''

new = '''  technicianSelect.addEventListener(
    "change",
    updateSubmitState
  );

  workOrderNumber.addEventListener(
    "input",
    updateSubmitState
  );
'''

if old not in text:
    raise SystemExit(
        "ERROR: technician event block not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Remove document click logic for W/O dropdown if present.
old = '''  document.addEventListener(
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


'''

text = text.replace(
    old,
    "",
    1,
)


# Replace submit validation.
old = '''      if (!woId.value) {

        event.preventDefault();

        alert(
          "SELECT WORK ORDER."
        );

        woSearch.focus();

        return;
      }
'''

new = '''      if (!workOrderNumber.value.trim()) {

        event.preventDefault();

        alert(
          "ENTER WORK ORDER #."
        );

        workOrderNumber.focus();

        return;
      }
'''

if old not in text:
    raise SystemExit(
        "ERROR: submit W/O validation not found"
    )

text = text.replace(
    old,
    new,
    1,
)


path.write_text(
    text,
    encoding="utf-8",
)

print("OK: appliance_issue_new.html now uses manual W/O")
