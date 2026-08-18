from __future__ import annotations

from datetime import datetime

from extensions import db
from models import (
    Permission,
    User,
    UserPermission,
    UserPermissionAudit,
    UserWarehouseAccess,
    Warehouse,
)


class AccessManagementError(Exception):
    pass


class AccessManagementService:
    """
    Write-side access management.

    IMPORTANT:
    Methods support commit=False so a caller can combine multiple access
    changes into one atomic transaction.

    UI bulk updates should use commit=False for every individual operation
    and call db.session.commit() once at the end.
    """

    @staticmethod
    def _finish(commit: bool) -> None:
        if commit:
            db.session.commit()
        else:
            db.session.flush()

    @staticmethod
    def create_warehouse(
        *,
        code: str,
        name: str,
        actor: User,
        description: str | None = None,
        commit: bool = True,
    ) -> Warehouse:
        code_clean = (code or "").strip().upper()
        name_clean = (name or "").strip()

        if not code_clean:
            raise AccessManagementError(
                "Warehouse code is required."
            )

        if not name_clean:
            raise AccessManagementError(
                "Warehouse name is required."
            )

        existing = (
            Warehouse.query
            .filter(
                db.or_(
                    Warehouse.code == code_clean,
                    Warehouse.name == name_clean,
                )
            )
            .first()
        )

        if existing:
            raise AccessManagementError(
                "Warehouse with this code or name already exists."
            )

        warehouse = Warehouse(
            code=code_clean,
            name=name_clean,
            description=(description or "").strip() or None,
            is_active=True,
            created_by_id=actor.id,
            updated_by_id=actor.id,
        )

        db.session.add(warehouse)
        db.session.flush()

        db.session.add(
            UserPermissionAudit(
                target_user_id=None,
                action="WAREHOUSE_CREATED",
                warehouse_id=warehouse.id,
                old_value=None,
                new_value=(
                    f"{warehouse.code} | "
                    f"{warehouse.name}"
                ),
                actor_user_id=actor.id,
                actor_username=actor.username,
            )
        )

        AccessManagementService._finish(commit)

        return warehouse

    @staticmethod
    def grant_warehouse_access(
        *,
        user: User,
        warehouse: Warehouse,
        actor: User,
        is_default: bool = False,
        commit: bool = True,
    ) -> UserWarehouseAccess:
        access = (
            UserWarehouseAccess.query
            .filter_by(
                user_id=user.id,
                warehouse_id=warehouse.id,
            )
            .first()
        )

        # If this warehouse becomes default, remove default flag
        # from every other warehouse for this user.
        if is_default:
            query = (
                UserWarehouseAccess.query
                .filter(
                    UserWarehouseAccess.user_id == user.id,
                )
            )

            if access is not None:
                query = query.filter(
                    UserWarehouseAccess.id != access.id
                )

            query.update(
                {"is_default": False},
                synchronize_session=False,
            )

        if access is None:
            access = UserWarehouseAccess(
                user_id=user.id,
                warehouse_id=warehouse.id,
                is_default=is_default,
                is_active=True,
                created_by_id=actor.id,
            )

            db.session.add(access)
            db.session.flush()

            db.session.add(
                UserPermissionAudit(
                    target_user_id=user.id,
                    action="WAREHOUSE_ACCESS_GRANTED",
                    warehouse_id=warehouse.id,
                    old_value=None,
                    new_value=(
                        f"active=True,"
                        f"default={is_default}"
                    ),
                    actor_user_id=actor.id,
                    actor_username=actor.username,
                )
            )

        else:
            old_active = bool(access.is_active)
            old_default = bool(access.is_default)

            new_active = True
            new_default = bool(is_default)

            # Do not create useless audit rows when nothing changed.
            changed = (
                old_active != new_active
                or old_default != new_default
            )

            access.is_active = new_active
            access.is_default = new_default

            if changed:
                db.session.add(
                    UserPermissionAudit(
                        target_user_id=user.id,
                        action="WAREHOUSE_ACCESS_UPDATED",
                        warehouse_id=warehouse.id,
                        old_value=(
                            f"active={old_active},"
                            f"default={old_default}"
                        ),
                        new_value=(
                            f"active={new_active},"
                            f"default={new_default}"
                        ),
                        actor_user_id=actor.id,
                        actor_username=actor.username,
                    )
                )

        AccessManagementService._finish(commit)

        return access

    @staticmethod
    def revoke_warehouse_access(
        *,
        user: User,
        warehouse: Warehouse,
        actor: User,
        commit: bool = True,
    ) -> None:
        access = (
            UserWarehouseAccess.query
            .filter_by(
                user_id=user.id,
                warehouse_id=warehouse.id,
            )
            .first()
        )

        if access is None or not access.is_active:
            return

        old_value = (
            f"active={bool(access.is_active)},"
            f"default={bool(access.is_default)}"
        )

        access.is_active = False
        access.is_default = False

        db.session.add(
            UserPermissionAudit(
                target_user_id=user.id,
                action="WAREHOUSE_ACCESS_REVOKED",
                warehouse_id=warehouse.id,
                old_value=old_value,
                new_value="active=False,default=False",
                actor_user_id=actor.id,
                actor_username=actor.username,
            )
        )

        AccessManagementService._finish(commit)

    @staticmethod
    def grant_permission(
        *,
        user: User,
        permission_code: str,
        actor: User,
        commit: bool = True,
    ) -> UserPermission:
        code = (permission_code or "").strip()

        permission = (
            Permission.query
            .filter_by(
                code=code,
                is_active=True,
            )
            .first()
        )

        if permission is None:
            raise AccessManagementError(
                f"Unknown permission: {code}"
            )

        existing = (
            UserPermission.query
            .filter_by(
                user_id=user.id,
                permission_id=permission.id,
            )
            .first()
        )

        if existing is not None:
            return existing

        grant = UserPermission(
            user_id=user.id,
            permission_id=permission.id,
            granted_by_id=actor.id,
            granted_at=datetime.utcnow(),
        )

        db.session.add(grant)

        db.session.add(
            UserPermissionAudit(
                target_user_id=user.id,
                action="PERMISSION_GRANTED",
                permission_code=permission.code,
                old_value="DENIED",
                new_value="ALLOWED",
                actor_user_id=actor.id,
                actor_username=actor.username,
            )
        )

        AccessManagementService._finish(commit)

        return grant

    @staticmethod
    def revoke_permission(
        *,
        user: User,
        permission_code: str,
        actor: User,
        commit: bool = True,
    ) -> None:
        permission = (
            Permission.query
            .filter_by(
                code=(permission_code or "").strip()
            )
            .first()
        )

        if permission is None:
            return

        grant = (
            UserPermission.query
            .filter_by(
                user_id=user.id,
                permission_id=permission.id,
            )
            .first()
        )

        if grant is None:
            return

        db.session.delete(grant)

        db.session.add(
            UserPermissionAudit(
                target_user_id=user.id,
                action="PERMISSION_REVOKED",
                permission_code=permission.code,
                old_value="ALLOWED",
                new_value="DENIED",
                actor_user_id=actor.id,
                actor_username=actor.username,
            )
        )

        AccessManagementService._finish(commit)
