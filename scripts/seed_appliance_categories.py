from __future__ import annotations

from extensions import db
from models import ApplianceCategory


APPLIANCE_CATEGORIES = [
    {
        "code": "REFRIGERATOR",
        "name": "Refrigerator",
        "sort_order": 10,
    },
    {
        "code": "FREEZER",
        "name": "Freezer",
        "sort_order": 20,
    },
    {
        "code": "WASHER",
        "name": "Washer",
        "sort_order": 30,
    },
    {
        "code": "DRYER",
        "name": "Dryer",
        "sort_order": 40,
    },
    {
        "code": "DISHWASHER",
        "name": "Dishwasher",
        "sort_order": 50,
    },
    {
        "code": "RANGE",
        "name": "Range",
        "sort_order": 60,
    },
    {
        "code": "OVEN",
        "name": "Oven",
        "sort_order": 70,
    },
    {
        "code": "COOKTOP",
        "name": "Cooktop",
        "sort_order": 80,
    },
    {
        "code": "MICROWAVE",
        "name": "Microwave",
        "sort_order": 90,
    },
    {
        "code": "HOOD",
        "name": "Range Hood",
        "sort_order": 100,
    },
    {
        "code": "WINE_COOLER",
        "name": "Wine Cooler",
        "sort_order": 110,
    },
    {
        "code": "ICE_MAKER",
        "name": "Ice Maker",
        "sort_order": 120,
    },
    {
        "code": "HVAC",
        "name": "HVAC",
        "sort_order": 200,
    },
    {
        "code": "OTHER",
        "name": "Other",
        "sort_order": 900,
    },
]


def seed_appliance_categories():
    created = 0
    updated = 0

    for data in APPLIANCE_CATEGORIES:
        category = (
            ApplianceCategory.query
            .filter_by(code=data["code"])
            .first()
        )

        if category is None:
            category = ApplianceCategory(
                code=data["code"],
                name=data["name"],
                sort_order=data["sort_order"],
                is_active=True,
            )

            db.session.add(category)
            created += 1
            continue

        changed = False

        if category.name != data["name"]:
            category.name = data["name"]
            changed = True

        if category.sort_order != data["sort_order"]:
            category.sort_order = data["sort_order"]
            changed = True

        if not category.is_active:
            category.is_active = True
            changed = True

        if changed:
            updated += 1

    db.session.commit()

    total = ApplianceCategory.query.count()

    print("APPLIANCE CATEGORY SEED OK")
    print("Created:", created)
    print("Updated:", updated)
    print("Total:", total)

    return {
        "created": created,
        "updated": updated,
        "total": total,
    }
