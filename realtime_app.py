from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cow_detection_secret'

# Prefer eventlet if available (optional). This doesn't force-install it.
try:
    import eventlet  # noqa: F401
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
except Exception:
    socketio = SocketIO(app, cors_allowed_origins="*")

# Globals for models (will be loaded lazily)
yolo_model = None
breed_model = None

breed_classes = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
    'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
    'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
    'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur'
]

# Basic logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


@app.before_first_request
def announce_startup():
    # This ensures Render logs show the service started and bound to a port.
    app.logger.info("Server starting up and ready to accept requests.")


@app.route('/')
def index():
    return render_template('realtime.html')


@app.route('/load_model', methods=['POST'])
def load_model():
    """
    Lazy-load heavy ML libraries and the models on demand.
    This prevents long startup times (TensorFlow, YOLO, matplotlib font cache)
    from blocking the server binding on Render.
    """
    global yolo_model, breed_model
    try:
        # Import heavy packages here to avoid blocking application startup
        from ultralytics import YOLO
        import tensorflow as tf

        # You may want to change these paths or fetch them at build time
        yolo_weights = request.json.get('yolo_weights', 'yolov8n.pt') if request.is_json else 'yolov8n.pt'
        breed_weights = request.json.get('breed_weights', 'classifier.h5') if request.is_json else 'classifier.h5'

        app.logger.info(f"Loading YOLO weights from: {yolo_weights}")
        yolo_model = YOLO(yolo_weights)

        app.logger.info(f"Loading breed classifier from: {breed_weights}")
        breed_model = tf.keras.models.load_model(breed_weights)

        return jsonify({'status': 'success', 'message': 'Models loaded successfully'})
    except Exception as e:
        app.logger.exception("Error loading models")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def classify_breed(cow_crop):
    """
    Classify cropped cow image using breed_model.
    Assumes breed_model has already been loaded via /load_model.
    """
    global breed_model
    img = cv2.resize(cow_crop, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0

    prediction = breed_model.predict(img, verbose=0)
    breed_idx = int(np.argmax(prediction))
    confidence = float(prediction[0][breed_idx])

    return breed_classes[breed_idx], confidence


frame_count = 0


@socketio.on('video_frame')
def handle_video_frame(data):
    global frame_count, yolo_model, breed_model
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

        # Run YOLO detection (class id 19 is used here)
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
                        cv2.putText(frame, label, (x1, y1 - 10),
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
        app.logger.exception("Error during frame handling")
        emit('detection_result', {'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Ensure the app binds to the environment port so Render detects it
    app.logger.info(f"Starting SocketIO server on 0.0.0.0:{port}")
    socketio.run(app, debug=False, host="0.0.0.0", port=port)
