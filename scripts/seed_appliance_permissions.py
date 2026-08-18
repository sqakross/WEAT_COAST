from __future__ import annotations

from extensions import db
from models import Permission


APPLIANCE_PERMISSIONS = [
    {
        "code": "appliance.view",
        "name": "View Appliance Inventory",
        "description": "View appliance inventory for allowed warehouses.",
        "group_name": "Appliance Inventory",
        "sort_order": 10,
    },
    {
        "code": "appliance.receive",
        "name": "Create Receiving",
        "description": "Create appliance receiving documents for allowed warehouses.",
        "group_name": "Receiving",
        "sort_order": 20,
    },
    {
        "code": "appliance.post_receiving",
        "name": "Post Receiving",
        "description": "Post appliance receiving documents and place received units into stock.",
        "group_name": "Receiving",
        "sort_order": 30,
    },
    {
        "code": "appliance.correct_own",
        "name": "Correct Own Operations",
        "description": "Correct eligible appliance operations created by the same user.",
        "group_name": "Receiving",
        "sort_order": 40,
    },
    {
        "code": "appliance.correct_others",
        "name": "Correct Other Users' Operations",
        "description": "Correct eligible appliance operations created by other users.",
        "group_name": "Receiving",
        "sort_order": 50,
    },
    {
        "code": "appliance.issue",
        "name": "Issue Appliance",
        "description": "Issue appliance units from allowed warehouses.",
        "group_name": "Stock Operations",
        "sort_order": 60,
    },
    {
        "code": "appliance.return",
        "name": "Return Appliance",
        "description": "Return appliance units into allowed warehouses.",
        "group_name": "Stock Operations",
        "sort_order": 70,
    },
    {
        "code": "appliance.transfer",
        "name": "Transfer Appliance",
        "description": "Transfer appliance units between allowed warehouses.",
        "group_name": "Stock Operations",
        "sort_order": 80,
    },
    {
        "code": "appliance.vendor_return",
        "name": "Vendor Return",
        "description": "Return appliance units to a vendor.",
        "group_name": "Disposition",
        "sort_order": 90,
    },
    {
        "code": "appliance.sell",
        "name": "Sell Appliance",
        "description": "Mark and process appliance units as sold.",
        "group_name": "Disposition",
        "sort_order": 100,
    },
    {
        "code": "appliance.write_off",
        "name": "Write Off Appliance",
        "description": "Write off or scrap appliance units.",
        "group_name": "Disposition",
        "sort_order": 110,
    },
    {
        "code": "appliance.pricing",
        "name": "Manage Appliance Pricing",
        "description": "Enter and update appliance cost on receiving documents.",
        "group_name": "Financial",
        "sort_order": 120,
    },
    {
        "code": "appliance.audit",
        "name": "View Appliance Audit",
        "description": "View appliance history and audit records for allowed warehouses.",
        "group_name": "Audit",
        "sort_order": 130,
    },
    {
        "code": "appliance.audit_full",
        "name": "View Full Warehouse Audit",
        "description": "View full appliance audit history for allowed warehouses.",
        "group_name": "Audit",
        "sort_order": 140,
    },
    {
        "code": "appliance.categories_manage",
        "name": "Manage Appliance Categories",
        "description": "Manage appliance categories and specification definitions.",
        "group_name": "Administration",
        "sort_order": 150,
    },
    {
        "code": "appliance.legacy_import",
        "name": "Legacy Excel Import",
        "description": "Import legacy appliance inventory from Excel or CSV.",
        "group_name": "Administration",
        "sort_order": 160,
    },
    {
        "code": "access.manage_users",
        "name": "Manage User Access",
        "description": "Manage user access configuration.",
        "group_name": "Access Administration",
        "sort_order": 170,
    },
    {
        "code": "access.manage_warehouses",
        "name": "Manage Warehouses",
        "description": "Create and manage warehouses.",
        "group_name": "Access Administration",
        "sort_order": 180,
    },
    {
        "code": "access.manage_permissions",
        "name": "Manage Permissions",
        "description": "Assign or remove user permissions and warehouse access.",
        "group_name": "Access Administration",
        "sort_order": 190,
    },
]


def seed_permissions() -> dict:
    created = 0
    updated = 0

    for item in APPLIANCE_PERMISSIONS:
        permission = Permission.query.filter_by(code=item["code"]).first()

        if permission is None:
            permission = Permission(
                code=item["code"],
                name=item["name"],
                description=item["description"],
                group_name=item["group_name"],
                is_active=True,
                sort_order=item["sort_order"],
            )
            db.session.add(permission)
            created += 1
            continue

        changed = False

        for field in (
            "name",
            "description",
            "group_name",
            "sort_order",
        ):
            new_value = item[field]

            if getattr(permission, field) != new_value:
                setattr(permission, field, new_value)
                changed = True

        if not permission.is_active:
            permission.is_active = True
            changed = True

        if changed:
            updated += 1

    db.session.commit()

    total = Permission.query.filter(
        Permission.code.in_(
            [item["code"] for item in APPLIANCE_PERMISSIONS]
        )
    ).count()

    result = {
        "created": created,
        "updated": updated,
        "total": total,
    }

    print("PERMISSION SEED OK")
    print("Created:", created)
    print("Updated:", updated)
    print("Total:", total)

    return result
