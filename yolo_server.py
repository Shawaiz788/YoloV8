"""
YOLO Fruit/Vegetable Detection Server
======================================
Endpoints:
  POST /predict               — Upload image → count, weight, crop, prices
  GET  /market/{crop}         — Live PSBA market prices for a crop
  GET  /market/all            — All cached prices across every category
  GET  /weight/{crop}/{count} — Weight estimate only
  GET  /cache/refresh         — Manually trigger a cache refresh

Setup:
  pip install fastapi uvicorn ultralytics pillow playwright beautifulsoup4
  playwright install chromium
"""

from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
import io
import asyncio
import time
import socket
import threading
import json
import os
from bs4 import BeautifulSoup
from PIL import Image
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

app = FastAPI()
model = YOLO('yolov8n.pt')


# ── Weight Estimation ─────────────────────────────────────────────────────────
WEIGHT_MULTIPLIERS = {
    "apple": 182, "orange": 154, "banana": 118, "mango": 310,
    "grape": 5, "watermelon": 9000, "pineapple": 900, "pear": 175,
    "peach": 150, "strawberry": 12, "lemon": 84, "lime": 67,
    "tomato": 123, "potato": 213, "onion": 110, "carrot": 61,
    "broccoli": 350, "cucumber": 300, "capsicum": 160,
    "cauliflower": 600, "cabbage": 900, "garlic": 60, "ginger": 100,
    "fruit": 150, "vegetable": 200,
}

def estimate_weight(crop: str, count: int) -> dict:
    grams_each = WEIGHT_MULTIPLIERS.get(crop, WEIGHT_MULTIPLIERS["fruit"])
    if count == 0:
        return {"min_kg": 0.0, "max_kg": 0.0, "avg_kg": 0.0, "display": "0 Kg"}
    min_kg = round((max(1, count - 1) * grams_each) / 1000, 2)
    max_kg = round(((count + 1)       * grams_each) / 1000, 2)
    avg_kg = round((count             * grams_each) / 1000, 2)
    return {"min_kg": min_kg, "max_kg": max_kg, "avg_kg": avg_kg,
            "display": f"{min_kg} – {max_kg} Kg  (avg {avg_kg} Kg)"}


# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict[str, list[dict]] = {}
_cache_timestamp: float = 0.0
_cache_lock = asyncio.Lock()
_CACHE_TTL = 30 * 60  # 30 minutes
_CACHE_FILE = "market_cache.json"


def _save_cache_to_file():
    """Persist cache to JSON file."""
    try:
        with open(_CACHE_FILE, 'w') as f:
            json.dump(_cache, f, indent=2)
        print(f"[Cache] Saved to {_CACHE_FILE}")
    except Exception as e:
        print(f"[Cache] Failed to save: {e}")


def _load_cache_from_file():
    """Load cache from JSON file if it exists."""
    global _cache, _cache_timestamp
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r') as f:
                _cache = json.load(f)
            _cache_timestamp = time.time()
            print(f"[Cache] Loaded {len(_cache)} items from {_CACHE_FILE}")
            return True
        except Exception as e:
            print(f"[Cache] Failed to load: {e}")
    return False


def _cache_is_fresh() -> bool:
    return bool(_cache) and (time.time() - _cache_timestamp) < _CACHE_TTL


def _lookup_from_cache(crop_name: str) -> list[dict]:
    """Search cache for any commodity containing crop_name as substring."""
    term = crop_name.strip().lower()
    results = []
    for key, records in _cache.items():
        if term in key:
            results.extend(records)
    return results


# ── PSBA Scraper ──────────────────────────────────────────────────────────────

def _parse_price(value: str) -> float | None:
    if not value:
        return None
    cleaned = value.replace(",", "").replace(" ", "").replace("Rs", "").replace("↓", "").replace("↑", "").split("%")[0].strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_table_html(html_fragment: str, category: str) -> list[dict]:
    """
    Parse the rendered HTML for ONE tab.
    Handles both <table> and div-based layouts that React apps use.
    Returns records with: commodity, dc_price, psba_price, discount, category
    """
    soup = BeautifulSoup(html_fragment, "html.parser")
    records = []

    # ── Strategy 1: real <table> element ──────────────────────────────────────
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        headers = [c.get_text(strip=True).lower() for c in header_cells]

        def col_idx(*keywords):
            for kw in keywords:
                for i, h in enumerate(headers):
                    if kw in h:
                        return i
            return None

        item_col     = col_idx("item", "name", "commodity", "product")
        dc_col       = col_idx("dc price", "dc", "market price", "original")
        psba_col     = col_idx("psba price", "psba", "sahulat", "discounted", "sale")
        discount_col = col_idx("discount", "saving", "diff")

        if item_col     is None: item_col     = 0
        if dc_col       is None: dc_col       = 1
        if psba_col     is None: psba_col     = 2
        if discount_col is None: discount_col = 3

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            def cell(idx):
                if idx is None or idx >= len(cells):
                    return ""
                for img in cells[idx].find_all("img"):
                    img.decompose()
                return cells[idx].get_text(strip=True)

            commodity = cell(item_col)
            if not commodity or commodity.lower() in ("item", "name", "commodity"):
                continue

            records.append({
                "commodity":  commodity,
                "city":       "Punjab",
                "mandi":      commodity,
                "price":      int(_parse_price(cell(psba_col)) or 0),
                "dc_price":   _parse_price(cell(dc_col)),
                "discount":   cell(discount_col),
                "category":   category,
                "unit":       "Kg",
            })

    if records:
        return records

    # ── Strategy 2: div/grid layout (no <table> tag) ──────────────────────────
    row_containers = (
        soup.select("[class*='row']:not(html):not(body)")
        or soup.select("[class*='item-row']")
        or soup.select("[class*='price-row']")
        or soup.select("[class*='list-item']")
    )
    for container in row_containers:
        cells = container.find_all(["div", "span", "td"], recursive=False)
        if len(cells) < 2:
            cells = container.find_all(["div", "span"])
        texts = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
        if len(texts) >= 2:
            records.append({
                "commodity":  texts[0],
                "city":       "Punjab",
                "mandi":      texts[0],
                "price":      int(_parse_price(texts[2]) if len(texts) > 2 else 0),
                "dc_price":   _parse_price(texts[1]) if len(texts) > 1 else None,
                "discount":   texts[3] if len(texts) > 3 else "",
                "category":   category,
                "unit":       "Kg",
            })

    return records


async def _scrape_all_tabs() -> dict[str, list[dict]]:
    """
    Launch browser, scrape every tab on the PSBA page, return full cache dict.
    Returns empty dict if scraping fails (will trigger fallback to saved cache).
    """
    url = "https://psba.gop.pk/daily-rate-comparison"
    all_records: list[dict] = []

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()

            print(f"[PSBA] Loading page...")
            try:
                await page.goto(url, wait_until="networkidle", timeout=30_000)
            except PlaywrightTimeout:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            try:
                await page.wait_for_selector("table, [class*='tab'], [role='tab'], button", timeout=15_000)
            except PlaywrightTimeout:
                print("[PSBA] Warning: initial content timeout.")

            await asyncio.sleep(2)

            TAB_SELECTORS = [
                "[role='tab']", ".nav-tabs .nav-link", ".tab-list button", "[class*='TabButton']",
                "[class*='tab-button']", "[class*='tab-item']", "[class*='TabItem']",
                "[class*='category-tab']", ".MuiTab-root", "[class*='ant-tabs-tab']",
                "nav button", "header button",
            ]

            tabs = []
            for sel in TAB_SELECTORS:
                found = await page.query_selector_all(sel)
                if len(found) >= 2:
                    tabs = found
                    break

            if not tabs:
                all_buttons = await page.query_selector_all("button")
                TAB_KEYWORDS = {"vegetables", "fruits", "grains", "meat", "eggs", "pulses", "dairy"}
                for btn in all_buttons:
                    txt = (await btn.inner_text()).strip().lower()
                    if any(kw in txt for kw in TAB_KEYWORDS):
                        tabs.append(btn)

            if tabs:
                for tab in tabs:
                    tab_name = (await tab.inner_text()).strip()
                    print(f"[PSBA] → Scraping tab: '{tab_name}'")
                    await tab.click()
                    try:
                        await page.wait_for_selector("table, [class*='row']", timeout=8_000, state="visible")
                    except PlaywrightTimeout:
                        pass
                    await asyncio.sleep(1.5)
                    html = await page.content()
                    all_records.extend(_parse_table_html(html, category=tab_name))
            else:
                print("[PSBA] No tabs found. Scraping single page.")
                html = await page.content()
                all_records = _parse_table_html(html, category="General")

            await browser.close()

        cache: dict[str, list[dict]] = {}
        for record in all_records:
            key = record["commodity"].strip().lower()
            if key not in cache: cache[key] = []
            cache[key].append(record)
        return cache
    
    except Exception as e:
        print(f"[PSBA] Scraping failed: {e}")
        return {}


async def ensure_cache_fresh():
    """Refresh the global cache if stale. Thread-safe via asyncio lock.
    Falls back to saved cache if scraping fails."""
    global _cache, _cache_timestamp
    if _cache_is_fresh(): return
    async with _cache_lock:
        if _cache_is_fresh(): return
        print("[Cache] Cache refreshing...")
        try:
            new_cache = await _scrape_all_tabs()
            if new_cache:
                _cache = new_cache
                _cache_timestamp = time.time()
                _save_cache_to_file()
                print("[Cache] Successfully updated and saved")
            else:
                print("[Cache] Scraping returned empty. Using saved cache...")
                _load_cache_from_file()
        except Exception as e:
            print(f"[Cache] Update failed: {e}. Using saved cache...")
            _load_cache_from_file()


@app.on_event("startup")
async def startup_event():
    print("[App] Starting up...")
    _load_cache_from_file()
    print("[App] Pre-warming cache...")
    asyncio.create_task(ensure_cache_fresh())


# ── Discovery Listener ────────────────────────────────────────────────────────

def start_discovery_listener():
    """Listens for the app's discovery message and replies with this PC's IP."""
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        udp_socket.bind(('', 8001))
    except Exception as e:
        print(f"[Discovery] Bind failed: {e}")
        return

    print("[Discovery] Waiting for phone to find me on port 8001...")
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data.decode() == "DISCOVER_CROP_SERVER":
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(("8.8.8.8", 80))
                    my_ip = s.getsockname()[0]
                except:
                    my_ip = socket.gethostbyname(socket.gethostname())
                finally:
                    s.close()
                
                print(f"[Discovery] Phone found me! Sending IP: {my_ip}")
                response = f"SERVER_IP:{my_ip}:8000"
                udp_socket.sendto(response.encode(), addr)
        except Exception: pass


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a fruit/vegetable image → count, weight, live prices, and est. total price."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model.predict(image, conf=0.25)

    count = len(results[0].boxes) if results else 0
    detected_crop = "fruit"
    if count > 0:
        class_id = int(results[0].boxes.cls[0].item())
        detected_crop = model.names[class_id].lower()

    count_display = "0" if count == 0 else f"{max(1, count - 1)} – {count + 1}"
    weight = estimate_weight(detected_crop, count)

    # Trigger cache refresh in background (don't wait for it)
    asyncio.create_task(ensure_cache_fresh())
    
    # Use whatever cache is available immediately
    market_data = _lookup_from_cache(detected_crop)

    est_price_display = "N/A"
    if market_data and count > 0:
        prices = [r["price"] for r in market_data if r.get("price") and r["price"] > 0]
        if prices:
            avg_market_price = sum(prices) / len(prices)
            p_min = round(weight["min_kg"] * avg_market_price)
            p_max = round(weight["max_kg"] * avg_market_price)
            est_price_display = f"Rs {p_min} – {p_max}"

    return {
        "crop":   detected_crop.capitalize(),
        "count":  count_display,
        "weight": weight["display"],
        "estPrice": est_price_display,
        "market_prices": market_data,
        "confidence": round(float(results[0].boxes.conf.mean()), 2) if count > 0 else 0.0,
        "status": "Success",
    }

@app.get("/market/all")
async def get_all_market():
    await ensure_cache_fresh()
    flat = [r for records in _cache.values() for r in records]
    return {"total": len(flat), "data": flat, "status": "Success"}


@app.get("/market/{crop}")
async def get_market(crop: str):
    await ensure_cache_fresh()
    data = _lookup_from_cache(crop)
    return {"crop": crop.capitalize(), "count": len(data), "data": data, "status": "Success"}


@app.get("/cache/refresh")
async def refresh_cache():
    global _cache_timestamp
    _cache_timestamp = 0
    await ensure_cache_fresh()
    return {"message": "Cache refreshed", "status": "Success"}


if __name__ == "__main__":
    threading.Thread(target=start_discovery_listener, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)