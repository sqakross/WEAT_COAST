from flask import Flask

original_register_blueprint = Flask.register_blueprint


def traced_register_blueprint(self, blueprint, *args, **kwargs):
    print(
        "REGISTER:",
        blueprint.name,
        "| prefix:",
        blueprint.url_prefix,
        "| app_id:",
        id(self),
    )

    result = original_register_blueprint(
        self,
        blueprint,
        *args,
        **kwargs,
    )

    print(
        "  AFTER:",
        sorted(self.blueprints.keys()),
    )

    return result


Flask.register_blueprint = traced_register_blueprint

print("=" * 70)
print("IMPORTING APP")
print("=" * 70)

import app

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("MODULE FILE:", app.__file__)
print("FINAL APP ID:", id(app.app))
print(
    "FINAL BLUEPRINTS:",
    sorted(app.app.blueprints.keys()),
)

print()
print("APPLIANCE ROUTES:")

found = False

for rule in app.app.url_map.iter_rules():
    if (
        "appliance" in rule.endpoint
        or "/appliances/" in str(rule.rule)
    ):
        found = True
        print(
            rule.endpoint,
            "->",
            rule.rule,
            sorted(rule.methods),
        )

if not found:
    print("NONE")
