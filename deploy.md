# BULLSEYE Deployment Guide

## GitHub Repository Setup

### 1. Initialize Git Repository
```bash
cd yolo_cow_detection
git init
git add .
git commit -m "Initial commit: BULLSEYE cow detection system"
```

### 2. Create GitHub Repository
1. Go to GitHub.com and create a new repository named `bullseye-cow-detection`
2. Add remote origin:
```bash
git remote add origin https://github.com/YOUR_USERNAME/bullseye-cow-detection.git
git branch -M main
git push -u origin main
```

## Local Deployment

### Prerequisites
- Python 3.8+
- Webcam (for real-time detection)
- `classifier.h5` model file in the root directory

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
python realtime_app.py
```
Access at: http://localhost:5000

## Cloud Deployment Options

### 1. Heroku Deployment
```bash
# Install Heroku CLI
# Create Procfile
echo "web: python realtime_app.py" > Procfile

# Deploy
heroku create bullseye-cow-detection
git push heroku main
```

### 2. Railway Deployment
1. Connect GitHub repository to Railway
2. Set environment variables if needed
3. Deploy automatically

### 3. Render Deployment
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python realtime_app.py`

## Model Files
- `classifier.h5`: Breed classification model (150x150 input)
- `yolov8n.pt`: YOLO detection model (auto-downloaded)

## Environment Variables (if needed)
```
FLASK_ENV=production
PORT=5000
```