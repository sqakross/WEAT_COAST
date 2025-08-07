from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os

load_dotenv()

MARCONE_LOGIN = os.getenv("MARCONE_LOGIN")
MARCONE_PASSWORD = os.getenv("MARCONE_PASSWORD")
LOGIN_URL = "https://my.marcone.com/"
CART_URL = "https://my.marcone.com/Cart"

def get_cart_items():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=30)  # Можно уменьшить до 30
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/Los_Angeles",
            java_script_enabled=True
        )
        page = context.new_page()

        # 🔐 Авторизация
        page.goto(LOGIN_URL)
        page.fill('input#UserName', MARCONE_LOGIN)
        page.wait_for_timeout(600)
        page.fill('input#Password', MARCONE_PASSWORD)
        page.wait_for_timeout(600)
        page.click('input#loginbtn')
        page.wait_for_timeout(600)

        # 🛒 Переход в корзину
        page.goto(CART_URL)
        page.wait_for_timeout(800)  # не меньше

        # 🧾 Сбор данных
        part_inputs = page.query_selector_all('input[name*=".Part"]')
        qty_selects = page.query_selector_all('select[name*="listCartItemByWareHouse"][name$=".Quantity"]')

        print(f"Найдено Part полей: {len(part_inputs)}")
        print(f"Найдено Qty полей: {len(qty_selects)}")

        cart_items = []
        for part_input, qty_select in zip(part_inputs, qty_selects):
            part_number = part_input.get_attribute("value") or ""
            qty_str = qty_select.input_value() or "0"
            try:
                qty = int(qty_str)
            except ValueError:
                qty = 0

            if part_number and qty:
                print(f"📦 Найдено: {part_number} | Qty: {qty}")
                cart_items.append({
                    "part_number": part_number.strip(),
                    "qty": qty
                })

        browser.close()
        return cart_items

# 🔍 Прямой запуск
if __name__ == "__main__":
    items = get_cart_items()
    for item in items:
        print(item)


