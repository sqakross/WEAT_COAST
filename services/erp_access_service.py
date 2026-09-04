from __future__ import annotations

from datetime import datetime

from extensions import db

from models import (
    ROLE_SUPERADMIN,
    ErpAccessOverride,
    User,
)

from services.erp_permission_registry import (
    ERP_PERMISSION_CODES,
    ERP_PERMISSION_GROUPS,
    get_erp_permission,
)


class ErpAccessError(Exception):
    pass


class ErpAccessService:
    """
    Compatibility layer for existing WCCR ERP authorization.

    Resolution:

        SUPERADMIN
            always allowed

        DENY override
            denied

        ALLOW override
            allowed

        no override
            existing hard-coded ERP result is used

    Therefore existing role behavior remains the default.
    """

    EFFECT_ALLOW = "ALLOW"
    EFFECT_DENY = "DENY"

    @staticmethod
    def is_superadmin(user: User | None) -> bool:

        if user is None:
            return False

        return (
            (getattr(user, "role", None) or "")
            .strip()
            .lower()
            == ROLE_SUPERADMIN
        )

    @staticmethod
    def _normalize_code(
        permission_code: str,
    ) -> str:

        code = (
            permission_code or ""
        ).strip()

        if code not in ERP_PERMISSION_CODES:
            raise ErpAccessError(
                f"Unknown ERP permission: {code}"
            )

        return code

    @classmethod
    def get_override(
        cls,
        user: User | None,
        permission_code: str,
    ) -> str | None:

        if user is None:
            return None

        code = cls._normalize_code(
            permission_code
        )

        if cls.is_superadmin(user):
            return cls.EFFECT_ALLOW

        row = (
            ErpAccessOverride.query
            .filter_by(
                user_id=user.id,
                permission_code=code,
            )
            .first()
        )

        if row is None:
            return None

        effect = (
            row.effect or ""
        ).strip().upper()

        if effect == cls.EFFECT_ALLOW:
            return cls.EFFECT_ALLOW

        if effect == cls.EFFECT_DENY:
            return cls.EFFECT_DENY

        return cls.EFFECT_DENY

    @classmethod
    def is_allowed(
        cls,
        user: User | None,
        permission_code: str,
        *,
        default_allowed: bool,
    ) -> bool:

        if user is None:
            return False

        if cls.is_superadmin(user):
            return True

        effect = cls.get_override(
            user,
            permission_code,
        )

        if effect == cls.EFFECT_DENY:
            return False

        if effect == cls.EFFECT_ALLOW:
            return True

        return bool(default_allowed)

    @classmethod
    def set_override(
        cls,
        *,
        user: User,
        permission_code: str,
        effect: str | None,
        actor: User,
        commit: bool = True,
    ):

        if user is None:
            raise ErpAccessError(
                "Target user is required."
            )

        if actor is None:
            raise ErpAccessError(
                "Actor is required."
            )

        if cls.is_superadmin(user):
            raise ErpAccessError(
                "Superadmin ERP access cannot be restricted."
            )

        code = cls._normalize_code(
            permission_code
        )

        normalized = (
            effect or ""
        ).strip().upper()

        if normalized in {
            "",
            "DEFAULT",
            "NONE",
        }:
            normalized = None

        if normalized not in {
            None,
            cls.EFFECT_ALLOW,
            cls.EFFECT_DENY,
        }:
            raise ErpAccessError(
                "Effect must be DEFAULT, ALLOW or DENY."
            )

        row = (
            ErpAccessOverride.query
            .filter_by(
                user_id=user.id,
                permission_code=code,
            )
            .first()
        )

        if normalized is None:

            if row is not None:
                db.session.delete(row)

            if commit:
                db.session.commit()
            else:
                db.session.flush()

            return None

        now = datetime.utcnow()

        if row is None:

            row = ErpAccessOverride(
                user_id=user.id,
                permission_code=code,
                effect=normalized,
                created_at=now,
                created_by_id=actor.id,
                updated_at=now,
                updated_by_id=actor.id,
            )

            db.session.add(row)

        else:

            row.effect = normalized
            row.updated_at = now
            row.updated_by_id = actor.id

        if commit:
            db.session.commit()
        else:
            db.session.flush()

        return row

    @classmethod
    def override_map(
        cls,
        user: User | None,
    ) -> dict[str, str]:

        if user is None:
            return {}

        if cls.is_superadmin(user):

            return {
                code: cls.EFFECT_ALLOW
                for code in ERP_PERMISSION_CODES
            }

        rows = (
            ErpAccessOverride.query
            .filter_by(
                user_id=user.id
            )
            .all()
        )

        return {
            row.permission_code: row.effect
            for row in rows
            if row.permission_code
            in ERP_PERMISSION_CODES
        }

    @classmethod
    def clear_user_overrides(
        cls,
        *,
        user: User,
        commit: bool = True,
    ) -> int:

        if user is None:
            return 0

        if cls.is_superadmin(user):
            return 0

        count = (
            ErpAccessOverride.query
            .filter_by(
                user_id=user.id
            )
            .delete(
                synchronize_session=False
            )
        )

        if commit:
            db.session.commit()
        else:
            db.session.flush()

        return int(count or 0)

    @staticmethod
    def registry():
        return ERP_PERMISSION_GROUPS

    @staticmethod
    def permission_info(
        permission_code: str,
    ):
        return get_erp_permission(
            permission_code
        )
