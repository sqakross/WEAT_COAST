from flask import Blueprint

supplier_returns_bp = Blueprint(
    "supplier_returns",
    __name__,
    template_folder="templates",
)

from . import routes  # noqa
