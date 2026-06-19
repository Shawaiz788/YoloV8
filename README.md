# YOLO Crop Detection Server

Fruit and vegetable detection with market price estimation using YOLO and PSBA scraping.

## 📁 Folder Structure

```
yolos/
├── yolo_server.py              # Local development (full-featured)
├── vercel/                      # Vercel deployment
│   ├── api/
│   │   └── index.py            # Serverless handler
│   ├── vercel.json             # Vercel config
│   └── requirements.txt         # Python dependencies
├── .gitignore                  # Git ignore rules
├── VERCEL_DEPLOYMENT.md        # Deployment guide
└── README.md                   # This file
```

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn ultralytics pillow playwright beautifulsoup4

# Install Playwright chromium
playwright install chromium

# Run server
python yolo_server.py
```

Access at: `http://localhost:8000`

### Vercel Deployment

```bash
cd vercel
vercel
```

## 📊 Key Differences

| Feature | Local | Vercel |
|---------|-------|--------|
| File persistence | ✅ | ❌ |
| Browser automation | ✅ | ❌ |
| Timeout | Unlimited | 60s |
| Cold start | N/A | 5-10s |

## 🔗 API Endpoints

### POST /predict
Upload an image to detect crops and get prices.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
```

### GET /market/{crop}
Get prices for a specific crop.

```bash
curl http://localhost:8000/market/apple
```

### GET /market/all
Get all cached prices.

```bash
curl http://localhost:8000/market/all
```

### GET /weight/{crop}/{count}
Get weight estimate.

```bash
curl http://localhost:8000/weight/apple/3
```

### GET /cache/refresh
Manually refresh cache.

```bash
curl http://localhost:8000/cache/refresh
```

## 📝 For Detailed Deployment Guide

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)
