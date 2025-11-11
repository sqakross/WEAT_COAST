from flask import Blueprint
from .routes import supplier_returns_bp


supplier_returns_bp = Blueprint(
    "supplier_returns",
    __name__,
    template_folder="templates",
)

from . import routes  # noqa
