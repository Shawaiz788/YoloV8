# YOLO Crop Detection API - Vercel Deployment Guide

## Project Structure

```
yolos/
├── api/
│   └── index.py           # Main API (Vercel serverless handler)
├── yolo_server.py         # Local development version
├── vercel.json            # Vercel config
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md
```

## What Changed for Vercel?

| Feature | Local | Vercel |
|---------|-------|--------|
| File persistence | ✅ JSON file cache | 🔄 In-memory (reset on cold start) |
| Browser automation | ✅ Playwright | ❌ Simple HTTP requests |
| Timeout | ❌ No limit | ⏱️ 60 seconds max |
| Memory | ❌ Unlimited | 🎯 3GB max |

## Deployment Steps

### 1. Prepare Repository

```bash
# Initialize git if not done
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git push origin main
```

### 2. Deploy to Vercel

**Option A: Using Vercel CLI**
```bash
npm install -g vercel
vercel
```

**Option B: Using Vercel Dashboard**
1. Go to https://vercel.com
2. Sign up / Log in
3. Click "New Project"
4. Select GitHub repo
5. Click "Deploy"

### 3. Configure Environment (if needed)

In Vercel Dashboard:
- Project Settings → Environment Variables
- Add any custom env vars (all will use defaults)

### 4. Get Your API URL

After deployment, Vercel gives you a URL like:
```
https://yolos.vercel.app/
```

## API Endpoints

### Health Check
```
GET /
```

### Predict (Main)
```
POST /predict
Content-Type: multipart/form-data

Body: file=<image_file>
```

Response:
```json
{
  "crop": "Apple",
  "count": "2 – 4",
  "weight": "0.35 – 0.55 Kg (avg 0.45 Kg)",
  "estPrice": "Rs 70 – 110",
  "market_prices": [...],
  "confidence": 0.89,
  "status": "Success"
}
```

### Get Market Prices for Crop
```
GET /market/{crop}

Example: /market/apple
```

### Get All Cached Prices
```
GET /market/all
```

### Manual Cache Refresh
```
GET /cache/refresh
```

### Weight Estimate Only
```
GET /weight/{crop}/{count}

Example: /weight/apple/3
```

## Frontend Integration

### Example with fetch:

```javascript
async function detectCrop(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('https://yolos.vercel.app/predict', {
    method: 'POST',
    body: formData,
  });
  
  const result = await response.json();
  return result;
}
```

### Using cURL:

```bash
curl -X POST https://yolos.vercel.app/predict \
  -F "file=@image.jpg"
```

## Performance Notes

⚡ **Cold Start**: First request may take 5-10s (model loading)
⚡ **Warm Start**: Subsequent requests ~1-2s (< 10min since last request)
⚡ **Cache**: In-memory, resets on cold start (acceptable for serverless)

## Limitations

❌ File persistence doesn't work (no permanent storage)
❌ Playwright/Browser automation not supported
❌ 60-second timeout per request
❌ Model re-downloads on cold start

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m uvicorn api.index:app --reload

# Access at http://localhost:8000
```

## Troubleshooting

**Model not loading?**
- Check Vercel logs: `vercel logs`
- Ensure yolov8n.pt is downloaded automatically

**Timeout errors?**
- Image detection takes 1-2s
- PSBA scraping can take 5-10s
- Keep within 60s limit

**Cache always empty?**
- Normal for serverless (resets on cold start)
- First request rebuilds cache
- Next requests use it

## Production Checklist

- [x] `vercel.json` configured
- [x] `requirements.txt` complete
- [x] `.gitignore` set up
- [ ] GitHub repo connected
- [ ] CORS properly configured
- [ ] Error handling tested

## Support

For issues:
1. Check Vercel logs: `vercel logs --follow`
2. Verify model download
3. Test endpoints locally first
