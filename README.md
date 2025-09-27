# BULLSEYE - AI-Powered Bovine Detection & Classification System

BULLSEYE integrates YOLOv8 for cow detection with a Hybrid DL based breed classifier for real-time cow detection and breed classification with a cutting-edge dark techie interface.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download YOLOv8 model on first run.

## Usage

### Web Applications (Recommended)

#### Real-time Flask App (Inspired by original)
```bash
python realtime_app.py
```
Then open http://localhost:5000 in your browser

#### Streamlit App (Easy to use)
```bash
streamlit run web_app.py
```
Then open http://localhost:8501 in your browser

### Command Line Options

#### Live Detection (Webcam)
```bash
python live_detection.py
```

#### Live Detection (Video File)
```bash
python live_detection.py --source path/to/video.mp4
```

#### Batch Processing
```bash
python batch_detection.py --input_dir path/to/images --output_dir results
```

## Features

- **Web-based UI**: Modern browser interface with real-time processing
- **Streamlit App**: Easy-to-use interface with image upload, webcam, and batch processing
- **Flask App**: Real-time webcam detection with WebSocket communication
- Real-time cow detection using YOLOv8
- Breed classification using pre-trained Hybrid DL model (150x150 input)
- Multiple cow detection and classification in single frame
- Support for webcam and video file input
- Live results display with breed confidence scores
- Batch processing for multiple images

## Model Files

- YOLOv8: Downloads automatically
- Breed Classifier: Uses `classifier.h5`

## Controls

- Press 'q' to quit live detection
- Adjust confidence thresholds in `cow_detector.py` if needed