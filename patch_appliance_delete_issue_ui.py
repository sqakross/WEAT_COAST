from pathlib import Path
from datetime import datetime

routes_path = Path("appliance/routes.py")
template_path = Path("templates/appliance_issue_detail.html")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# BACKUPS
# ============================================================

for path in [routes_path, template_path]:

    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found"
        )

    backup = Path(
        str(path)
        + f".before_delete_issue_ui_{stamp}.bak"
    )

    backup.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print("BACKUP:", backup)


# ============================================================
# ROUTE
# ============================================================

routes_text = routes_path.read_text(
    encoding="utf-8"
)

if "def issue_delete(" not in routes_text:

    route_block = r'''

# ============================================================
# DELETE APPLIANCE ISSUE
#
# Superadmin-only correction tool.
#
# This is NOT a normal warehouse return.
# It removes an incorrectly created Issue only while the
# appliances are still untouched after the original ISSUE.
# ============================================================

@appliance_bp.post(
    "/issues/<int:issue_id>/delete"
)
@login_required
def issue_delete(issue_id):

    from models import (
        ApplianceIssue,
        ApplianceIssueLine,
        ApplianceMovement,
        ApplianceUnit,
    )

    # --------------------------------------------------------
    # Role protection
    # --------------------------------------------------------

    if (
        getattr(current_user, "role", "")
        or ""
    ).strip().lower() != "superadmin":

        flash(
            "Only SUPERADMIN can delete an Appliance Issue.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.issue_detail",
                issue_id=issue_id,
            )
        )

    issue = db.session.get(
        ApplianceIssue,
        issue_id,
    )

    if issue is None:

        flash(
            "Appliance Issue not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    # User must also have warehouse access.
    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=issue.warehouse_id,
    ):

        flash(
            "You do not have access to this warehouse.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    try:

        # ----------------------------------------------------
        # Lines / physical units
        # ----------------------------------------------------

        lines = (
            ApplianceIssueLine.query
            .filter(
                ApplianceIssueLine.issue_id
                == issue.id
            )
            .order_by(
                ApplianceIssueLine.line_no.asc()
            )
            .all()
        )

        if not lines:

            raise ApplianceIssueError(
                "Issue contains no appliance lines."
            )

        unit_ids = [
            int(line.appliance_unit_id)
            for line in lines
        ]

        units = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.id.in_(
                    unit_ids
                )
            )
            .all()
        )

        units_by_id = {
            int(unit.id): unit
            for unit in units
        }

        if len(units_by_id) != len(
            set(unit_ids)
        ):

            raise ApplianceIssueError(
                "One or more appliance units "
                "from this Issue no longer exist."
            )

        # ----------------------------------------------------
        # Find original ISSUE movements belonging to this AIS.
        # ----------------------------------------------------

        original_issue_movements = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.issue_id
                == issue.id,
                ApplianceMovement.movement_type
                == "ISSUE",
            )
            .all()
        )

        original_movement_ids = [
            movement.id
            for movement
            in original_issue_movements
        ]

        # ----------------------------------------------------
        # SAFETY CHECK:
        #
        # If ANY other movement already exists for one of these
        # physical appliances, normal DELETE is no longer valid.
        #
        # Future examples:
        # RETURN
        # REPLACE
        # CHANGE_WO
        # INSTALLED
        # EXCHANGE
        # ----------------------------------------------------

        later_query = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.appliance_unit_id.in_(
                    unit_ids
                )
            )
        )

        if original_movement_ids:

            later_query = later_query.filter(
                ~ApplianceMovement.id.in_(
                    original_movement_ids
                )
            )

        other_movement = (
            later_query
            .order_by(
                ApplianceMovement.created_at.desc()
            )
            .first()
        )

        if other_movement is not None:

            raise ApplianceIssueError(
                "This Issue cannot be deleted because "
                "one or more appliances already have "
                "additional movement history. "
                "Use RETURN / REPLACE / CHANGE W/O instead."
            )

        # ----------------------------------------------------
        # Current-state validation
        # ----------------------------------------------------

        for line in lines:

            unit = units_by_id[
                int(line.appliance_unit_id)
            ]

            status = (
                unit.status
                or ""
            ).strip().lower()

            if status != "issued":

                raise ApplianceIssueError(
                    f"{unit.inventory_number} is currently "
                    f"{status.upper() or 'UNKNOWN'}. "
                    "Issue deletion is no longer allowed."
                )

            current_wo = (
                unit.current_work_order_number
                or ""
            ).strip().upper()

            issue_wo = (
                issue.work_order_number
                or ""
            ).strip().upper()

            if (
                current_wo
                and issue_wo
                and current_wo != issue_wo
            ):

                raise ApplianceIssueError(
                    f"{unit.inventory_number} is already "
                    "assigned to another Work Order. "
                    "Issue deletion is blocked."
                )

        issue_number = issue.issue_number

        # ----------------------------------------------------
        # Restore inventory
        # ----------------------------------------------------

        now = datetime.utcnow()

        for unit in units:

            unit.status = "available"

            unit.current_work_order_id = None
            unit.current_work_order_number = None

            unit.updated_at = now
            unit.updated_by_id = current_user.id

        db.session.flush()

        # ----------------------------------------------------
        # Remove original ISSUE movements
        # ----------------------------------------------------

        (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.issue_id
                == issue.id
            )
            .delete(
                synchronize_session=False
            )
        )

        # ----------------------------------------------------
        # Remove Issue Lines
        # ----------------------------------------------------

        (
            ApplianceIssueLine.query
            .filter(
                ApplianceIssueLine.issue_id
                == issue.id
            )
            .delete(
                synchronize_session=False
            )
        )

        db.session.flush()

        # ----------------------------------------------------
        # Remove Issue header
        #
        # Bulk delete avoids SQLAlchemy trying to cascade-delete
        # already deleted lines a second time.
        # ----------------------------------------------------

        (
            ApplianceIssue.query
            .filter(
                ApplianceIssue.id
                == issue.id
            )
            .delete(
                synchronize_session=False
            )
        )

        db.session.commit()

        flash(
            f"{issue_number} deleted. "
            f"{len(units)} appliance(s) returned to AVAILABLE.",
            "success",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    except (
        ApplianceIssueError,
        ValueError,
    ) as exc:

        db.session.rollback()

        flash(
            str(exc),
            "danger",
        )

    except Exception:

        db.session.rollback()
        raise

    return redirect(
        url_for(
            "appliance.issue_detail",
            issue_id=issue_id,
        )
    )

'''

    routes_text = (
        routes_text.rstrip()
        + "\n"
        + route_block
        + "\n"
    )

    routes_path.write_text(
        routes_text,
        encoding="utf-8",
    )

    print("OK: issue_delete route added")

else:

    print(
        "SKIP: issue_delete route already exists"
    )


# ============================================================
# TEMPLATE BUTTON
# ============================================================

template_text = template_path.read_text(
    encoding="utf-8"
)

if "DELETE ISSUE" not in template_text:

    needle = '''      <button type="button"
              class="btn btn-success btn-sm"
              onclick="window.print();">
        Print
      </button>
'''

    replacement = '''      <button type="button"
              class="btn btn-success btn-sm"
              onclick="window.print();">
        Print
      </button>

      {% if current_user.role == 'superadmin' %}

        <form method="post"
              action="{{ url_for(
                  'appliance.issue_delete',
                  issue_id=issue.id
              ) }}"
              class="d-inline"
              onsubmit="return confirm(
                'DELETE {{ issue.issue_number }}?\\n\\n' +
                'Technician: {{ issue.technician_username|upper }}\\n' +
                'W/O: {{ issue.work_order_number|upper }}\\n' +
                'Units: {{ issue.appliance_count }}\\n\\n' +
                'All appliances will return to AVAILABLE.\\n' +
                'This action is only for an incorrectly created Issue.'
              );">

          <button type="submit"
                  class="btn btn-outline-danger btn-sm">
            DELETE ISSUE
          </button>

        </form>

      {% endif %}
'''

    if needle not in template_text:

        raise SystemExit(
            "ERROR: Print button block not found "
            "in appliance_issue_detail.html"
        )

    template_text = template_text.replace(
        needle,
        replacement,
        1,
    )

    template_path.write_text(
        template_text,
        encoding="utf-8",
    )

    print("OK: DELETE ISSUE button added")

else:

    print(
        "SKIP: DELETE ISSUE button already exists"
    )


print()
print("=" * 70)
print("DELETE ISSUE UI PATCH COMPLETE")
print("=" * 70)
