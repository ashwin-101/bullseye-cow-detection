from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import tensorflow as tf
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cow_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

yolo_model = None
breed_model = None
breed_classes = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']

@app.route('/')
def index():
    return render_template('realtime.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    global yolo_model, breed_model
    try:
        yolo_model = YOLO('yolov8n.pt')
        breed_model = tf.keras.models.load_model('classifier.h5')
        return jsonify({'status': 'success', 'message': 'Models loaded successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def classify_breed(cow_crop):
    img = cv2.resize(cow_crop, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0
    
    prediction = breed_model.predict(img, verbose=0)
    breed_idx = np.argmax(prediction)
    confidence = prediction[0][breed_idx]
    
    return breed_classes[breed_idx], confidence

frame_count = 0

@socketio.on('video_frame')
def handle_video_frame(data):
    global frame_count
    if yolo_model is None or breed_model is None:
        emit('detection_result', {'status': 'error', 'message': 'Models not loaded'})
        return
    
    frame_count += 1
    # Process every 2nd frame to reduce lag
    if frame_count % 2 != 0:
        return
    
    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        detections = yolo_model(frame, classes=[19], verbose=False)[0]
        results = []
        
        if detections.boxes is not None:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.5:
                    cow_crop = frame[y1:y2, x1:x2]
                    
                    if cow_crop.size > 0:
                        breed, breed_conf = classify_breed(cow_crop)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"{breed}: {breed_conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        results.append({
                            'breed': breed,
                            'breed_conf': float(breed_conf),
                            'detection_conf': confidence
                        })
        
        # Compress image more for faster transmission
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('detection_result', {
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_base64}",
            'results': results
        })
    except Exception as e:
        emit('detection_result', {'status': 'error', 'message': str(e)})
"""
if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
"""
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000)) 
    socketio.run(app, debug=False, host="0.0.0.0", port=port)
