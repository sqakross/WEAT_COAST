from pathlib import Path

path = Path("templates/users.html")
text = path.read_text(encoding="utf-8")

backup = Path("templates/users.html.before_access_ui")
backup.write_text(text, encoding="utf-8")

anchor = """                  {% elif r == 'admin' %}
                    <span class="badge rounded-pill"
                          style="background:#0d6efd; font-size:.7rem; font-weight:600; padding:.4rem .6rem;">
                      ADMIN
                    </span>
"""

replacement = """                  {% elif r == 'manager' %}
                    <span class="badge rounded-pill"
                          style="background:#6f42c1; font-size:.7rem; font-weight:600; padding:.4rem .6rem;">
                      MANAGER
                    </span>

                  {% elif r == 'admin' %}
                    <span class="badge rounded-pill"
                          style="background:#0d6efd; font-size:.7rem; font-weight:600; padding:.4rem .6rem;">
                      ADMIN
                    </span>
"""

if anchor not in text:
    raise SystemExit("ERROR: admin badge anchor not found")

text = text.replace(anchor, replacement, 1)


anchor = """                        <!-- Edit -->
                        <a class="btn btn-light btn-sm border shadow-sm"
"""

replacement = """                        {% if (current_user.role or '')|lower == 'superadmin' %}
                        <a class="btn btn-outline-dark btn-sm shadow-sm"
                           style="font-weight:500; line-height:1.2; border-radius:.6rem; min-width:82px;"
                           href="{{ url_for('inventory.user_access', user_id=u.id) }}">
                          ⚙️ Access
                        </a>
                        {% endif %}

                        <!-- Edit -->
                        <a class="btn btn-light btn-sm border shadow-sm"
"""

if anchor not in text:
    raise SystemExit("ERROR: Edit button anchor not found")

text = text.replace(anchor, replacement, 1)


old = """            Roles define access to inventory / issue / receiving screens.
"""

new = """            Role defines general ERP access. Appliance warehouse access and permissions can be configured separately.
"""

if old not in text:
    raise SystemExit("ERROR: user list footer not found")

text = text.replace(old, new, 1)


old = """            superadmin = full access, admin = manage users & stock,
            technician = issue parts, viewer = read-only.
"""

new = """            superadmin = global access, manager/admin = configurable access,
            technician = operational user, viewer = read-only.
"""

if old not in text:
    raise SystemExit("ERROR: create-card footer not found")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("OK: templates/users.html patched")
print("Backup:", backup)
