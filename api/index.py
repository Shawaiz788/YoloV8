"""
YOLO Fruit/Vegetable Detection Server (Vercel Serverless)
===========================================================
Optimized for Vercel deployment with serverless architecture
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import io
import asyncio
import os
import json
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model only once per container
try:
    model = YOLO('yolov8n.pt')
except:
    model = None

# In-memory cache (resets on cold start, but that's ok)
_cache = {}
_cache_timestamp = 0


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


# ── PSBA Scraper (Simplified for Serverless) ──────────────────────────────────

def _parse_price(value: str) -> float | None:
    if not value:
        return None
    cleaned = value.replace(",", "").replace(" ", "").replace("Rs", "").replace("↓", "").replace("↑", "").split("%")[0].strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _scrape_psba_prices() -> dict[str, list[dict]]:
    """
    Scrape PSBA using simple HTTP requests (no browser needed).
    More reliable for serverless environment.
    """
    try:
        url = "https://psba.gop.pk/daily-rate-comparison"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        
        # Parse tables
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            
            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    commodity = cells[0].get_text(strip=True)
                    if commodity and commodity.lower() not in ("item", "name", "commodity"):
                        records.append({
                            "commodity": commodity,
                            "city": "Punjab",
                            "mandi": commodity,
                            "price": int(_parse_price(cells[2].get_text(strip=True)) or 0),
                            "dc_price": _parse_price(cells[1].get_text(strip=True)) if len(cells) > 1 else None,
                            "category": "General",
                            "unit": "Kg",
                        })
        
        # Convert to cache format
        cache = {}
        for record in records:
            key = record["commodity"].strip().lower()
            if key not in cache:
                cache[key] = []
            cache[key].append(record)
        
        return cache if cache else {}
    
    except Exception as e:
        print(f"[PSBA] Scraping failed: {e}")
        return {}


def _lookup_from_cache(crop_name: str) -> list[dict]:
    """Search cache for crop by name."""
    term = crop_name.strip().lower()
    results = []
    for key, records in _cache.items():
        if term in key:
            results.extend(records)
    return results


def _refresh_cache():
    """Update cache from PSBA (non-blocking in prediction flow)."""
    global _cache, _cache_timestamp
    try:
        new_cache = _scrape_psba_prices()
        if new_cache:
            _cache = new_cache
            _cache_timestamp = 1
            print("[Cache] Updated successfully")
    except Exception as e:
        print(f"[Cache] Refresh failed: {e}")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "YOLO Crop Detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload image → count, weight, crop, prices."""
    if not model:
        return {"error": "Model not loaded", "status": "Error"}
    
    try:
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
        
        # Refresh cache in background if empty (don't block)
        if not _cache:
            _refresh_cache()
        
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
            "crop": detected_crop.capitalize(),
            "count": count_display,
            "weight": weight["display"],
            "estPrice": est_price_display,
            "market_prices": market_data,
            "confidence": round(float(results[0].boxes.conf.mean()), 2) if count > 0 else 0.0,
            "status": "Success",
        }
    except Exception as e:
        return {"error": str(e), "status": "Error"}


@app.get("/market/{crop}")
def get_market(crop: str):
    """Get cached prices for a crop."""
    if not _cache:
        _refresh_cache()
    
    data = _lookup_from_cache(crop)
    return {
        "crop": crop.capitalize(),
        "count": len(data),
        "data": data,
        "status": "Success"
    }


@app.get("/market/all")
def get_all_market():
    """Get all cached prices."""
    if not _cache:
        _refresh_cache()
    
    flat = [r for records in _cache.values() for r in records]
    return {"total": len(flat), "data": flat, "status": "Success"}


@app.get("/cache/refresh")
def refresh_cache_endpoint():
    """Manually trigger cache refresh."""
    _refresh_cache()
    return {"message": "Cache refresh initiated", "status": "Success"}


@app.get("/weight/{crop}/{count}")
def get_weight(crop: str, count: int):
    """Get weight estimate only."""
    weight = estimate_weight(crop, count)
    return {"crop": crop, "count": count, "weight": weight, "status": "Success"}


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
