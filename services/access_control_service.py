from __future__ import annotations

from collections.abc import Iterable

from models import (
    ROLE_SUPERADMIN,
    Permission,
    User,
    UserPermission,
    UserWarehouseAccess,
    Warehouse,
)


class AccessControlService:
    """
    Central access-control service.

    Role answers WHO the user is.
    Warehouse access answers WHERE the user may work.
    Permission grants answer WHAT the user may do.

    Superadmin bypasses warehouse and permission restrictions.
    """

    @staticmethod
    def is_superadmin(user: User | None) -> bool:
        if user is None:
            return False

        return (
            (getattr(user, "role", None) or "").strip().lower()
            == ROLE_SUPERADMIN
        )

    @classmethod
    def has_permission(
        cls,
        user: User | None,
        permission_code: str,
    ) -> bool:
        if user is None:
            return False

        if cls.is_superadmin(user):
            return True

        code = (permission_code or "").strip()

        if not code:
            return False

        grant = (
            UserPermission.query
            .join(
                Permission,
                UserPermission.permission_id == Permission.id,
            )
            .filter(
                UserPermission.user_id == user.id,
                Permission.code == code,
                Permission.is_active.is_(True),
            )
            .first()
        )

        return grant is not None

    @classmethod
    def has_any_permission(
        cls,
        user: User | None,
        permission_codes: Iterable[str],
    ) -> bool:
        if user is None:
            return False

        if cls.is_superadmin(user):
            return True

        codes = {
            (code or "").strip()
            for code in permission_codes
            if (code or "").strip()
        }

        if not codes:
            return False

        grant = (
            UserPermission.query
            .join(
                Permission,
                UserPermission.permission_id == Permission.id,
            )
            .filter(
                UserPermission.user_id == user.id,
                Permission.code.in_(codes),
                Permission.is_active.is_(True),
            )
            .first()
        )

        return grant is not None

    @classmethod
    def has_all_permissions(
        cls,
        user: User | None,
        permission_codes: Iterable[str],
    ) -> bool:
        if user is None:
            return False

        if cls.is_superadmin(user):
            return True

        codes = {
            (code or "").strip()
            for code in permission_codes
            if (code or "").strip()
        }

        if not codes:
            return True

        granted_codes = {
            row[0]
            for row in (
                UserPermission.query
                .join(
                    Permission,
                    UserPermission.permission_id == Permission.id,
                )
                .with_entities(Permission.code)
                .filter(
                    UserPermission.user_id == user.id,
                    Permission.code.in_(codes),
                    Permission.is_active.is_(True),
                )
                .all()
            )
        }

        return codes.issubset(granted_codes)

    @classmethod
    def has_warehouse_access(
        cls,
        user: User | None,
        warehouse_id: int | None,
    ) -> bool:
        if user is None or warehouse_id is None:
            return False

        if cls.is_superadmin(user):
            warehouse = Warehouse.query.get(warehouse_id)

            return bool(
                warehouse is not None
                and warehouse.is_active
            )

        access = (
            UserWarehouseAccess.query
            .filter(
                UserWarehouseAccess.user_id == user.id,
                UserWarehouseAccess.warehouse_id == warehouse_id,
                UserWarehouseAccess.is_active.is_(True),
            )
            .first()
        )

        if access is None:
            return False

        return bool(
            access.warehouse is not None
            and access.warehouse.is_active
        )

    @classmethod
    def can(
        cls,
        user: User | None,
        permission_code: str,
        warehouse_id: int | None = None,
    ) -> bool:
        """
        Main authorization check.

        Without warehouse_id:
            checks only permission.

        With warehouse_id:
            requires BOTH permission and access to that warehouse.

        Superadmin bypasses explicit permission grants but an explicitly
        referenced warehouse still has to exist and be active.
        """
        if user is None:
            return False

        if not cls.has_permission(user, permission_code):
            return False

        if warehouse_id is None:
            return True

        return cls.has_warehouse_access(
            user,
            warehouse_id,
        )

    @classmethod
    def accessible_warehouses(
        cls,
        user: User | None,
    ) -> list[Warehouse]:
        if user is None:
            return []

        if cls.is_superadmin(user):
            return (
                Warehouse.query
                .filter(Warehouse.is_active.is_(True))
                .order_by(Warehouse.name.asc())
                .all()
            )

        return (
            Warehouse.query
            .join(
                UserWarehouseAccess,
                UserWarehouseAccess.warehouse_id == Warehouse.id,
            )
            .filter(
                UserWarehouseAccess.user_id == user.id,
                UserWarehouseAccess.is_active.is_(True),
                Warehouse.is_active.is_(True),
            )
            .order_by(
                UserWarehouseAccess.is_default.desc(),
                Warehouse.name.asc(),
            )
            .all()
        )

    @classmethod
    def default_warehouse(
        cls,
        user: User | None,
    ) -> Warehouse | None:
        if user is None:
            return None

        if cls.is_superadmin(user):
            return None

        access = (
            UserWarehouseAccess.query
            .join(
                Warehouse,
                UserWarehouseAccess.warehouse_id == Warehouse.id,
            )
            .filter(
                UserWarehouseAccess.user_id == user.id,
                UserWarehouseAccess.is_default.is_(True),
                UserWarehouseAccess.is_active.is_(True),
                Warehouse.is_active.is_(True),
            )
            .first()
        )

        if access is not None:
            return access.warehouse

        warehouses = cls.accessible_warehouses(user)

        return warehouses[0] if warehouses else None

    @classmethod
    def permission_codes(
        cls,
        user: User | None,
    ) -> set[str]:
        if user is None:
            return set()

        if cls.is_superadmin(user):
            return {
                code
                for (code,) in (
                    Permission.query
                    .with_entities(Permission.code)
                    .filter(Permission.is_active.is_(True))
                    .all()
                )
            }

        return {
            code
            for (code,) in (
                UserPermission.query
                .join(
                    Permission,
                    UserPermission.permission_id == Permission.id,
                )
                .with_entities(Permission.code)
                .filter(
                    UserPermission.user_id == user.id,
                    Permission.is_active.is_(True),
                )
                .all()
            )
        }
