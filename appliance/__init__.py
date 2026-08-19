from flask import Blueprint

appliance_bp = Blueprint(
    "appliance",
    __name__,
    url_prefix="/appliances",
)

from appliance import routes  # noqa: E402, F401
