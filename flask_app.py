from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json
from PIL import Image
import io
from cow_detector import CowDetectorClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cow_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

detector = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    global detector
    try:
        model_path = "classifier.h5"
        detector = CowDetectorClassifier(model_path)
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'})
    
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(file.stream)
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Process image
        results = detector.process_frame(img_array)
        output_img = detector.draw_results(img_array.copy(), results)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': img_base64,
            'results': results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@socketio.on('video_frame')
def handle_video_frame(data):
    if detector is None:
        emit('detection_result', {'status': 'error', 'message': 'Model not loaded'})
        return
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        results = detector.process_frame(frame)
        output_frame = detector.draw_results(frame.copy(), results)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', output_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('detection_result', {
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_base64}",
            'results': results
        })
    except Exception as e:
        emit('detection_result', {'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)