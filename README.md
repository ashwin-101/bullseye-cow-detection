# 🐂 BULLSEYE  
### AI-Powered Real-Time Bovine Detection & Breed Classification  

> ⚡ A next-gen AI system combining **YOLOv8 detection** + **Hybrid Deep Learning breed classification**, built for precision livestock management in India.  

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-BULLSEYE-brightgreen)](https://bullseye-live-cattle-detection.onrender.com/)  
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)  
[![Framework Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)  
[![Framework Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)  
[![Model YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)  

---

## 🌐 Live Deployment
🔴 **Try it right now:**  
👉 [BULLSEYE Live Cattle Detection](https://bullseye-live-cattle-detection.onrender.com/)  

Runs directly in your browser with **real-time cow detection & breed recognition**.

---

## ✨ Features

- 🖥 **Web-based Interface**: Real-time detection in the browser  
- 🤖 **YOLOv8-powered detection**: State-of-the-art object detection for cows  
- 🧬 **Hybrid DL Breed Classifier**: Classifies 40+ Indian cattle & buffalo breeds  
- 🎥 **Multiple Input Modes**: Webcam, video file, or image batch  
- 📊 **Explainable AI**: LIME & Grad-CAM ready for model interpretability  
- 🖼 **Dark Tech UI**: Modern interface for immersive experience  
- ⚡ **Optimized**: Lightweight deployment, works on Render Cloud  

---

## 🛠 Setup (Local Development)

1. Clone the repository:
 ```bash
 git clone https://github.com/ashwin-101/bullseye-cow-detection.git
 cd bullseye-cow-detection
 ```

2. Install dependencies:
 ```bash
 pip install -r requirements.txt
 ```

3. Run the Flask App:
 ```bash
 python realtime_app.py
 ```

Open http://localhost:5000
 in your browser.

Or launch the Streamlit App:
 ```bash
 streamlit run web_app.py
 ```

Open http://localhost:8501
 in your browser.


🎮 Usage
🔴 Real-Time Flask App
  ```bash
  python realtime_app.py
  ```

➡ Browser auto-loads realtime.html UI for live detection.

🟢 Streamlit App
 ```bash
 streamlit run web_app.py
 ```

➡ Upload images or use webcam easily.

📹 Live Detection (Webcam)
 ```bash
 python live_detection.py
 ```
🎞 Live Detection (Video File)
 ```bash
 python live_detection.py --source path/to/video.mp4
 ```
📁 Batch Processing
 ```bash
 python batch_detection.py --input_dir path/to/images --output_dir results
 ```

System Architecture (High-Level)
```
 ┌────────────┐       ┌──────────────┐       ┌─────────────┐
 │  Camera /  │       │   YOLOv8     │       │  Breed      │
 │  Upload    ├──────▶│  Cow Detector├──────▶│ Classifier  │
 └────────────┘       └──────────────┘       └─────────────┘
        │                       │                      │
        └─────► WebSocket ──────┴───────► Real-time UI │
```

⭐ If you like this project, don’t forget to star the repo on GitHub!


---
