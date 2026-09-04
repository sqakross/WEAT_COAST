from __future__ import annotations


"""
ERP permission registry.

This registry belongs to the EXISTING WCCR ERP.

It is intentionally separate from Appliance Inventory permissions.

Important:
    Defining a permission here DOES NOT enforce it automatically.

    Existing ERP role logic remains unchanged until a route is
    deliberately connected to ErpAccessService.
"""


ERP_PERMISSION_GROUPS = [

    # ========================================================
    # INVENTORY
    # ========================================================
    {
        "key": "inventory",
        "label": "Inventory",
        "description": "Parts inventory and stock operations.",
        "permissions": [
            {
                "code": "erp.inventory.access",
                "name": "Access Inventory",
                "description": "Open and use the Inventory module.",
            },
            {
                "code": "erp.inventory.part_create",
                "name": "Create Parts",
                "description": "Create new inventory parts.",
            },
            {
                "code": "erp.inventory.part_edit",
                "name": "Edit Parts",
                "description": "Edit existing inventory parts.",
            },
            {
                "code": "erp.inventory.part_delete",
                "name": "Delete Parts",
                "description": "Delete parts where ERP rules permit.",
            },
            {
                "code": "erp.inventory.issue",
                "name": "Issue Parts",
                "description": "Issue parts from inventory.",
            },
        ],
    },

    # ========================================================
    # WORK ORDERS
    # ========================================================
    {
        "key": "work_orders",
        "label": "Work Orders",
        "description": "Work order operations.",
        "permissions": [
            {
                "code": "erp.work_orders.access",
                "name": "Access Work Orders",
                "description": "Open and use Work Orders.",
            },
            {
                "code": "erp.work_orders.create",
                "name": "Create Work Orders",
                "description": "Create new work orders.",
            },
            {
                "code": "erp.work_orders.edit",
                "name": "Edit Work Orders",
                "description": "Edit work orders.",
            },
            {
                "code": "erp.work_orders.cancel",
                "name": "Cancel Work Orders",
                "description": "Cancel jobs where ERP rules permit.",
            },
            {
                "code": "erp.work_orders.issue_parts",
                "name": "Issue Parts",
                "description": "Issue parts directly from a work order.",
            },
        ],
    },

    # ========================================================
    # RECEIVING
    # ========================================================
    {
        "key": "receiving",
        "label": "Receiving",
        "description": "Receiving and posted receiving administration.",
        "permissions": [
            {
                "code": "erp.receiving.access",
                "name": "Access Receiving",
                "description": "Open the Receiving module.",
            },
            {
                "code": "erp.receiving.create",
                "name": "Create Receiving",
                "description": "Create receiving batches.",
            },
            {
                "code": "erp.receiving.post",
                "name": "Post Receiving",
                "description": "Post receiving into inventory.",
            },
            {
                "code": "erp.receiving.unpost",
                "name": "Unpost Receiving",
                "description": "Unpost receiving where ERP rules permit.",
            },
            {
                "code": "erp.receiving.edit_posted",
                "name": "Edit Posted Receiving",
                "description": "Edit posted receiving where ERP rules permit.",
            },
        ],
    },

    # ========================================================
    # RETURNS
    # ========================================================
    {
        "key": "returns",
        "label": "Returns",
        "description": "Supplier return operations.",
        "permissions": [
            {
                "code": "erp.returns.access",
                "name": "Access Returns",
                "description": "Open the Returns module.",
            },
            {
                "code": "erp.returns.create",
                "name": "Create Returns",
                "description": "Create supplier returns.",
            },
            # ERP RETURNS STEP 1B-v4 - PERMISSIONS
            {
                "code": "erp.returns.edit",
                "name": "Edit Returns",
                "description": "Edit supplier return headers and item rows.",
            },
            {
                "code": "erp.returns.post",
                "name": "Post Returns",
                "description": "Post supplier returns and decrement inventory.",
            },
            {
                "code": "erp.returns.unpost",
                "name": "Unpost Returns",
                "description": "Unpost supplier returns and restore inventory.",
            },
            {
                "code": "erp.returns.delete",
                "name": "Delete Returns",
                "description": "Delete draft supplier return documents.",
            },
        ],
    },

    # ========================================================
    # REPORTS
    # ========================================================
    {
        "key": "reports",
        "label": "Reports",
        "description": "ERP reporting and exports.",
        "permissions": [
            {
                "code": "erp.reports.access",
                "name": "Access Reports",
                "description": "Open ERP reports.",
            },
            {
                "code": "erp.reports.export",
                "name": "Export Reports",
                "description": "Export ERP reports.",
            },
            {
                "code": "erp.reports.view_costs",
                "name": "View Costs",
                "description": "View cost information in reports.",
            },
        ],
    },

    # ========================================================
    # TOOLS / ASSETS
    # ========================================================
    {
        "key": "tools",
        "label": "Tools / Assets",
        "description": "Tools and asset management.",
        "permissions": [
            {
                "code": "erp.tools.access",
                "name": "Access Tools / Assets",
                "description": "Open Tools / Assets.",
            },
            {
                "code": "erp.tools.assign",
                "name": "Assign Tools",
                "description": "Assign tools and assets.",
            },
            {
                "code": "erp.tools.transfer",
                "name": "Transfer Tools",
                "description": "Transfer tools between holders.",
            },
            {
                "code": "erp.tools.manage",
                "name": "Manage Tools / Assets",
                "description": "Administrative tool operations.",
            },
        ],
    },

    # ========================================================
    # ALERTS
    # ========================================================
    {
        "key": "alerts",
        "label": "Alerts",
        "description": "Operational alerts.",
        "permissions": [
            {
                "code": "erp.alerts.access",
                "name": "Access Alerts",
                "description": "Open and use Alerts.",
            },
        ],
    },

    # ========================================================
    # ACCOUNTING
    # ========================================================
    {
        "key": "accounting",
        "label": "Accounting",
        "description": "Accounting module.",
        "permissions": [
            {
                "code": "erp.accounting.access",
                "name": "Access Accounting",
                "description": "Open Accounting.",
            },
            {
                "code": "erp.accounting.manage",
                "name": "Manage Accounting",
                "description": "Perform accounting administration.",
            },
        ],
    },

    # ========================================================
    # USERS / SECURITY
    # ========================================================
    {
        "key": "users",
        "label": "Users / Security",
        "description": "User and access administration.",
        "permissions": [
            {
                "code": "erp.users.access",
                "name": "Access Manage Users",
                "description": "Open Manage Users.",
            },
            {
                "code": "erp.users.create",
                "name": "Create Users",
                "description": "Create users where role rules permit.",
            },
            {
                "code": "erp.users.edit",
                "name": "Edit Users",
                "description": "Edit users where role rules permit.",
            },
            {
                "code": "erp.users.change_role",
                "name": "Change User Roles",
                # ERP ACCESS USERS STEP 2 FIX02V2 - ROLE HELP
                "description": (
                    "ALLOW: Admin can create and assign the ADMIN role. "
                    "Technician, User, Viewer, Accounting, and Manager "
                    "do not require this elevated permission. "
                    "SUPERADMIN can only be assigned or modified by a Superadmin."
                ),
            },
            # ERP ACCESS USERS STEP 1 FINAL FIX02 - DELETE PERMISSION
            {
                "code": "erp.users.delete",
                "name": "Delete Users",
                "description": "Delete users where ERP rules permit.",
            },
            {
                "code": "erp.users.manage_access",
                "name": "Manage ERP Access",
                "description": "Configure ERP access overrides.",
            },
        ],
    },

    # ========================================================
    # ISSUED INVOICES
    # ========================================================
    {
        "key": "invoices",
        "label": "Issued Parts / Invoices",
        "description": "Issued-parts invoice administration.",
        "permissions": [
            {
                "code": "erp.invoices.access",
                "name": "Access Issued Invoices",
                "description": "Open issued-parts invoice views.",
            },
            {
                "code": "erp.invoices.delete",
                "name": "Delete Invoice",
                "description": "Delete an invoice where ERP rules permit.",
            },
        ],
    },

]


ERP_PERMISSION_INDEX = {
    permission["code"]: {
        **permission,
        "module_key": group["key"],
        "module_label": group["label"],
    }
    for group in ERP_PERMISSION_GROUPS
    for permission in group["permissions"]
}


ERP_PERMISSION_CODES = frozenset(ERP_PERMISSION_INDEX.keys())


def get_erp_permission(code: str) -> dict | None:
    return ERP_PERMISSION_INDEX.get(
        (code or "").strip()
    )


def is_erp_permission(code: str) -> bool:
    return (
        (code or "").strip()
        in ERP_PERMISSION_CODES
    )
