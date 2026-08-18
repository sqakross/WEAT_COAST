from pathlib import Path

path = Path("templates/user_access.html")
text = path.read_text(encoding="utf-8")

start_marker = '''      <div class="table-responsive">
        <table class="table table-sm align-middle mb-0"
               style="font-size:.78rem;">'''

start = text.find(start_marker)

if start == -1:
    raise SystemExit("ERROR: audit table start not found")

end_marker = '''      </div>

    </div>

  </div>
</div>
{% endblock %}'''

end = text.find(end_marker, start)

if end == -1:
    raise SystemExit("ERROR: audit table end not found")

new_table = '''      <div class="table-responsive">
        <table class="table table-sm align-middle mb-0"
               style="font-size:.78rem;">

          <thead class="table-light">
            <tr>
              <th style="min-width:150px;">Date</th>
              <th>Summary</th>
              <th style="min-width:110px;">Changed By</th>
              <th class="text-end" style="width:90px;">Details</th>
            </tr>
          </thead>

          <tbody>

            {% for group in recent_audit_groups %}
              {% set gid = loop.index %}

              <tr>
                <td class="text-nowrap">
                  {% if group.created_at_local %}
                    {{ group.created_at_local.strftime('%m/%d/%Y %I:%M %p') }}
                  {% else %}
                    —
                  {% endif %}
                </td>

                <td>
                  <div class="fw-semibold">
                    Access updated
                  </div>

                  <div class="text-muted"
                       style="font-size:.7rem;">

                    {% if group.permission_granted %}
                      <span class="me-2">
                        +{{ group.permission_granted|length }} permission{% if group.permission_granted|length != 1 %}s{% endif %}
                      </span>
                    {% endif %}

                    {% if group.permission_revoked %}
                      <span class="me-2">
                        -{{ group.permission_revoked|length }} permission{% if group.permission_revoked|length != 1 %}s{% endif %}
                      </span>
                    {% endif %}

                    {% if group.warehouse_granted %}
                      <span class="me-2">
                        +{{ group.warehouse_granted|length }} warehouse
                      </span>
                    {% endif %}

                    {% if group.warehouse_revoked %}
                      <span class="me-2">
                        -{{ group.warehouse_revoked|length }} warehouse
                      </span>
                    {% endif %}

                    {% if group.warehouse_updated %}
                      <span class="me-2">
                        {{ group.warehouse_updated|length }} warehouse updated
                      </span>
                    {% endif %}

                    {% if group.other %}
                      <span>
                        {{ group.other|length }} other change{% if group.other|length != 1 %}s{% endif %}
                      </span>
                    {% endif %}

                  </div>
                </td>

                <td>
                  {{ group.actor_username or '—' }}
                </td>

                <td class="text-end">
                  <button class="btn btn-light btn-sm border"
                          type="button"
                          data-bs-toggle="collapse"
                          data-bs-target="#auditDetails{{ gid }}"
                          aria-expanded="false"
                          style="border-radius:.5rem; font-size:.7rem;">
                    Details
                  </button>
                </td>
              </tr>

              <tr class="collapse"
                  id="auditDetails{{ gid }}">
                <td colspan="4"
                    style="background:#fafafa;">

                  <div class="p-2"
                       style="font-size:.72rem;">

                    {% if group.permission_granted %}
                      <div class="mb-2">
                        <div class="fw-semibold">
                          Permissions added
                        </div>
                        {% for code in group.permission_granted %}
                          <code class="d-block">{{ code }}</code>
                        {% endfor %}
                      </div>
                    {% endif %}

                    {% if group.permission_revoked %}
                      <div class="mb-2">
                        <div class="fw-semibold">
                          Permissions removed
                        </div>
                        {% for code in group.permission_revoked %}
                          <code class="d-block">{{ code }}</code>
                        {% endfor %}
                      </div>
                    {% endif %}

                    {% if group.warehouse_granted %}
                      <div class="mb-2">
                        <div class="fw-semibold">
                          Warehouse access added
                        </div>
                        {% for code in group.warehouse_granted %}
                          <code class="d-block">{{ code }}</code>
                        {% endfor %}
                      </div>
                    {% endif %}

                    {% if group.warehouse_revoked %}
                      <div class="mb-2">
                        <div class="fw-semibold">
                          Warehouse access removed
                        </div>
                        {% for code in group.warehouse_revoked %}
                          <code class="d-block">{{ code }}</code>
                        {% endfor %}
                      </div>
                    {% endif %}

                    {% if group.warehouse_updated %}
                      <div class="mb-2">
                        <div class="fw-semibold">
                          Warehouse access updated
                        </div>
                        {% for code in group.warehouse_updated %}
                          <code class="d-block">{{ code }}</code>
                        {% endfor %}
                      </div>
                    {% endif %}

                    {% if group.other %}
                      <div>
                        <div class="fw-semibold">
                          Other changes
                        </div>

                        {% for item in group.other %}
                          <div>
                            {{ item.action }}
                            {% if item.permission_code %}
                              · <code>{{ item.permission_code }}</code>
                            {% endif %}
                            {% if item.warehouse %}
                              · {{ item.warehouse }}
                            {% endif %}
                          </div>
                        {% endfor %}
                      </div>
                    {% endif %}

                  </div>

                </td>
              </tr>

            {% else %}

              <tr>
                <td colspan="4"
                    class="text-center text-muted py-3">
                  No access changes recorded yet.
                </td>
              </tr>

            {% endfor %}

          </tbody>
        </table>
      </div>

'''

text = text[:start] + new_table + text[end:]

path.write_text(text, encoding="utf-8")

print("OK: grouped access audit UI installed")
