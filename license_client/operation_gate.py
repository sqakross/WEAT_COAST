from __future__ import annotations

from license_client.runtime import (
    RuntimeMode,
    get_license_runtime,
)


class OperationAuthorizationError(RuntimeError):
    """Current application state does not permit protected mutations."""


def authorize_mutation(scope: str) -> None:
    """
    Local service-level authorization gate.

    E1/E2/E3 establish and refresh process authorization.
    E4 independently protects selected business mutations.

    No network access is performed here so this check is safe
    to execute inside existing database transaction boundaries.
    """
    runtime = get_license_runtime()
    snapshot = runtime.snapshot()

    if snapshot.mode == RuntimeMode.COLD:
        raise OperationAuthorizationError(
            "Application authorization has not been established."
        )

    if not snapshot.valid:
        raise OperationAuthorizationError(
            "Application authorization does not permit this operation."
        )

    if not scope or not scope.strip():
        raise OperationAuthorizationError(
            "Operation authorization scope is missing."
        )
