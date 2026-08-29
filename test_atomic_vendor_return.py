import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from app import app
from extensions import db
from models import ApplianceUnit
from services.appliance_issue_service import (
    ApplianceIssueError,
    ApplianceIssueService,
)


actor = SimpleNamespace(
    id=999999,
    role="superadmin",
)

fake_unit = SimpleNamespace(
    id=999999,
    warehouse_id=123,
    status="vendor_return_pending",
    inventory_number="TEST-ATOMIC-VR",
    stock_class="new",
    condition="new",
    current_work_order_id=None,
    current_work_order_number=None,
)

fake_warehouse = SimpleNamespace(
    id=123,
    code="TEST",
    is_active=True,
)


class FakeQuery:
    def __init__(self):
        self.update_calls = 0
        self.update_values = None

    def filter(self, *args, **kwargs):
        return self

    def update(
        self,
        values,
        synchronize_session=False,
    ):
        self.update_calls += 1
        self.update_values = values
        return 1


with app.app_context():

    # ========================================================
    # TEST 1
    # Valid Vendor Return metadata must be present in the
    # movement BEFORE the single commit.
    # ========================================================

    fake_query = FakeQuery()
    added_objects = []

    get_calls = {"count": 0}

    def fake_get(model, object_id):
        assert model is ApplianceUnit
        assert object_id == fake_unit.id

        get_calls["count"] += 1

        # First GET = source unit.
        # Second GET = refreshed unit after commit.
        return fake_unit

    def fake_add(obj):
        added_objects.append(obj)

    extra_meta = {
        "vendor_name": "TEST VENDOR",
        "rma_reference": "RMA-TEST-123",
        "return_notes": "Atomic metadata test",
        "source_receiving_number": "ARCV-TEST",
        "source_invoice_number": "INV-TEST",
    }

    with (
        patch.object(
            ApplianceIssueService,
            "_require_user",
            return_value=actor,
        ),
        patch.object(
            ApplianceIssueService,
            "_get_warehouse",
            return_value=fake_warehouse,
        ),
        patch(
            "services.appliance_issue_service.AccessControlService.can",
            return_value=True,
        ),
        patch.object(
            db.session,
            "get",
            side_effect=fake_get,
        ),
        patch.object(
            db.session,
            "query",
            return_value=fake_query,
        ),
        patch.object(
            db.session,
            "add",
            side_effect=fake_add,
        ),
        patch.object(
            db.session,
            "commit",
        ) as commit_mock,
        patch.object(
            db.session,
            "rollback",
        ) as rollback_mock,
    ):

        result = (
            ApplianceIssueService
            .change_inventory_disposition(
                actor=actor,
                appliance_unit_id=fake_unit.id,
                action="CONFIRM_VENDOR_RETURN",
                reason_code="RETURNED_TO_VENDOR",
                notes="Vendor: TEST VENDOR",
                extra_movement_meta=extra_meta,
            )
        )

        assert result is fake_unit

        assert fake_query.update_calls == 1, (
            "Expected exactly one inventory UPDATE"
        )

        assert len(added_objects) == 1, (
            "Expected exactly one movement to be added"
        )

        movement = added_objects[0]

        assert (
            movement.movement_type
            == "VENDOR_RETURN"
        )

        meta = json.loads(
            movement.meta_json
        )

        # Business metadata.
        for key, expected in extra_meta.items():
            actual = meta.get(key)

            assert actual == expected, (
                f"{key}: expected {expected!r}, "
                f"got {actual!r}"
            )

        # Service-owned audit metadata must remain present.
        assert (
            meta["inventory_number"]
            == "TEST-ATOMIC-VR"
        )

        assert (
            meta["action"]
            == "CONFIRM_VENDOR_RETURN"
        )

        assert (
            meta["from_status"]
            == "vendor_return_pending"
        )

        assert (
            meta["to_status"]
            == "vendor_return"
        )

        assert (
            meta["warehouse_id"]
            == 123
        )

        assert (
            meta["warehouse_code"]
            == "TEST"
        )

        assert (
            meta["reason_code"]
            == "RETURNED_TO_VENDOR"
        )

        commit_mock.assert_called_once()
        rollback_mock.assert_not_called()

    print()
    print("ATOMIC VENDOR RETURN TEST")
    print("-" * 72)
    print(
        "PASS: valid Vendor Return metadata was placed "
        "in movement before commit"
    )
    print(
        "PASS: exactly one inventory UPDATE was attempted"
    )
    print(
        "PASS: exactly one movement was created"
    )
    print(
        "PASS: exactly one commit was executed"
    )

    # ========================================================
    # TEST 2
    # Invalid JSON metadata must fail before UPDATE / ADD /
    # COMMIT.
    # ========================================================

    invalid_query = FakeQuery()

    with (
        patch.object(
            ApplianceIssueService,
            "_require_user",
            return_value=actor,
        ),
        patch.object(
            ApplianceIssueService,
            "_get_warehouse",
            return_value=fake_warehouse,
        ),
        patch(
            "services.appliance_issue_service.AccessControlService.can",
            return_value=True,
        ),
        patch.object(
            db.session,
            "get",
            return_value=fake_unit,
        ),
        patch.object(
            db.session,
            "query",
            return_value=invalid_query,
        ),
        patch.object(
            db.session,
            "add",
        ) as add_mock,
        patch.object(
            db.session,
            "commit",
        ) as commit_mock,
        patch.object(
            db.session,
            "rollback",
        ) as rollback_mock,
    ):

        try:
            (
                ApplianceIssueService
                .change_inventory_disposition(
                    actor=actor,
                    appliance_unit_id=fake_unit.id,
                    action="CONFIRM_VENDOR_RETURN",
                    reason_code="RETURNED_TO_VENDOR",
                    extra_movement_meta={
                        "bad_value": {1, 2, 3},
                    },
                )
            )

            raise AssertionError(
                "Invalid metadata unexpectedly succeeded"
            )

        except ApplianceIssueError as exc:
            assert (
                "not JSON serializable"
                in str(exc)
            )

        assert invalid_query.update_calls == 0, (
            "Inventory UPDATE occurred before metadata validation"
        )

        add_mock.assert_not_called()
        commit_mock.assert_not_called()

    print(
        "PASS: invalid metadata was rejected before inventory UPDATE"
    )
    print(
        "PASS: invalid metadata created no movement"
    )
    print(
        "PASS: invalid metadata executed no commit"
    )
    print(
        "PASS: no production ApplianceUnit was used"
    )
    print("-" * 72)
    print("ALL ATOMIC VENDOR RETURN TESTS PASSED")
