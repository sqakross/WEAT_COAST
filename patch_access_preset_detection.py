from pathlib import Path

path = Path("templates/user_access.html")
text = path.read_text(encoding="utf-8")

old = """  permissionBoxes().forEach(box => {
    box.addEventListener('change', () => {
      if (presetSelect) {
        presetSelect.value = 'custom';
      }
    });
  });
"""

new = """  function currentPermissionSet() {
    return new Set(
      permissionBoxes()
        .filter(box => box.checked)
        .map(box => box.value)
    );
  }


  function setsEqual(a, b) {
    if (a.size !== b.size) {
      return false;
    }

    for (const value of a) {
      if (!b.has(value)) {
        return false;
      }
    }

    return true;
  }


  function detectCurrentPreset() {
    if (!presetSelect) {
      return;
    }

    const current = currentPermissionSet();

    for (const [name, permissionSet] of Object.entries(presets)) {
      if (setsEqual(current, permissionSet)) {
        presetSelect.value = name;
        return;
      }
    }

    presetSelect.value = 'custom';
  }


  permissionBoxes().forEach(box => {
    box.addEventListener('change', () => {
      detectCurrentPreset();
    });
  });


  detectCurrentPreset();
"""

if old not in text:
    raise SystemExit("ERROR: preset change block not found")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("OK: preset auto-detection installed")
