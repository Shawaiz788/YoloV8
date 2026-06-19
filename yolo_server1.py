from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
import io
import requests
from bs4 import BeautifulSoup
from PIL import Image
from datetime import datetime

app = FastAPI()
model = YOLO('yolov8n.pt') 

# --- WEIGHT ESTIMATION MULTIPLIERS (Grams per fruit) ---
FRUIT_MULTIPLIERS = {
    "apple": 180,
    "orange": 150,
    "banana": 120,
    "mango": 300,
    "fruit": 150  # Default
}

def get_live_market_data(crop_name):
    # ... (Keep your existing scraper function here) ...
    return [] # Placeholder - keep your full scraper logic from the previous message

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model.predict(image, conf=0.25) 
    
    count = len(results[0].boxes) if results else 0
    detected_crop = "fruit"
    
    if count > 0:
        class_id = int(results[0].boxes.cls[0].item())
        detected_crop = model.names[class_id].lower()

    # 1. Calculate Count Range
    min_c = max(0, count - 1) if count > 0 else 0
    max_c = count + 1 if count > 0 else 0
    count_display = "0" if count == 0 else f"{min_c} - {max_c}"

    # 2. Calculate Weight Range (using Multiplier)
    multiplier = FRUIT_MULTIPLIERS.get(detected_crop, FRUIT_MULTIPLIERS["fruit"])
    
    if count == 0:
        weight_display = "0 Kg"
    else:
        # Convert grams to Kg for display
        min_w = (min_c * multiplier) / 1000.0
        max_w = (max_c * multiplier) / 1000.0
        weight_display = f"{min_w:.1f} - {max_w:.1f} Kg"

    return {
        "crop": detected_crop.capitalize(),
        "count": count_display,
        "weight": weight_display, # New field
        "confidence": 0.9,
        "status": "Success"
    }

@app.get("/market/{crop}")
async def get_market(crop: str):
    live_data = get_live_market_data(crop)
    return {"crop": crop.capitalize(), "data": live_data, "status": "Success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
