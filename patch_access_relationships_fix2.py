from pathlib import Path

path = Path("models.py")
text = path.read_text(encoding="utf-8")

# We only modify relationships inside the Access Control Foundation section.
start_marker = "# Access Control Foundation"
end_marker = "class JobReservation"

start = text.find(start_marker)
end = text.find(end_marker, start)

if start == -1:
    raise SystemExit("ERROR: Access Control Foundation section not found")

if end == -1:
    raise SystemExit("ERROR: end of Access Control Foundation not found")

before = text[:start]
section = text[start:end]
after = text[end:]

joined_count = section.count('lazy="joined"')
selectin_count = section.count('lazy="selectin"')

print("Before patch:")
print("  lazy=joined:", joined_count)
print("  lazy=selectin:", selectin_count)

section = section.replace(
    'lazy="joined"',
    'lazy="select"',
)

section = section.replace(
    'lazy="selectin"',
    'lazy="select"',
)

text = before + section + after

path.write_text(text, encoding="utf-8")

print()
print("OK: Access Control relationships changed to lazy='select'")
print("Changed joined relationships:", joined_count)
print("Changed remaining selectin relationships:", selectin_count)
