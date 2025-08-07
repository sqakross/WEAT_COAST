from playwright.sync_api import sync_playwright
import re

LOGIN = "099011"
PASSWORD = "PeYTfv5p^oQRgZv"
LOGIN_URL = "https://reliableparts.net/us/content/#/login"
CART_URL = "https://reliableparts.net/us/content/#/cart"

def get_cart_items():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=30)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/Los_Angeles",
            java_script_enabled=True
        )
        page = context.new_page()

        # Логин
        page.goto(LOGIN_URL)
        page.fill('input[name="userID"]', LOGIN)
        page.fill('input[type="password"]', PASSWORD)
        page.get_by_role("button", name="SIGN IN").click()
        page.wait_for_timeout(2000)

        # Корзина
        page.goto(CART_URL)
        print("✅ Перешли на корзину:", page.url)
        page.wait_for_load_state('networkidle')
        page.wait_for_function("document.readyState === 'complete'")
        page.wait_for_selector('a.item-name', timeout=10000)

        # 🔍 Найти все ссылки с номерами деталей
        part_elements = page.query_selector_all('a.item-name')
        print(f"🔎 Найдено Part полей: {len(part_elements)}")

        cart_items = []
        seen = set()

        for part_el in part_elements:
            raw_part = part_el.inner_text().strip()
            print(f"🧪 Raw part: '{raw_part}'")

            # Универсальный захват part number
            match = re.findall(r'[A-Z0-9\-]{5,}', raw_part)
            if not match:
                print(f"❌ Part number not matched: '{raw_part}'")
                continue

            part_number = match[-1]  # берём последнее "слово", которое подходит

            if part_number.lower() in ["sacramento", "none", "change"] or part_number in seen:
                continue

            # Найдём количество в том же блоке
            parent = part_el.evaluate_handle("el => el.closest('tr') || el.closest('div')")
            qty_el = parent.query_selector('input[type="text"]') if parent else None

            qty_str = qty_el.input_value().strip() if qty_el else "0"
            try:
                qty = int(qty_str)
            except ValueError:
                qty = 0

            if qty > 0:
                print(f"📦 Найдено: {part_number} | Qty: {qty}")
                seen.add(part_number)
                cart_items.append({
                    "part_number": part_number,
                    "qty": qty
                })

        browser.close()
        return cart_items

if __name__ == "__main__":
    items = get_cart_items()
    for item in items:
        print(item)
