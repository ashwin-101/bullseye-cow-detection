import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import torch

class CowDetectorClassifier:
    def __init__(self, breed_model_path, yolo_model='yolov8n.pt'):
        # Load YOLOv8 for detection
        self.yolo_model = YOLO(yolo_model)
        
        # Load breed classification model
        self.breed_model = tf.keras.models.load_model(breed_model_path)
        
        # Breed classes from the original model
        self.breed_classes = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']
        
    def detect_cows(self, frame):
        """Detect cows using YOLO"""
        results = self.yolo_model(frame, classes=[19])  # Class 19 is 'cow' in COCO
        return results[0]
    
    def classify_breed(self, cow_crop):
        """Classify cow breed using VGG16 model"""
        # Resize to model input size (150x150)
        img = cv2.resize(cow_crop, (150, 150))
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0
        
        # Predict breed
        prediction = self.breed_model.predict(img, verbose=0)
        breed_idx = np.argmax(prediction)
        confidence = prediction[0][breed_idx]
        
        return self.breed_classes[breed_idx], confidence
    
    def process_frame(self, frame):
        """Process single frame for detection and classification"""
        # Detect cows
        detections = self.detect_cows(frame)
        
        results = []
        
        if detections.boxes is not None:
            for box in detections.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.5:  # Confidence threshold
                    # Crop cow region
                    cow_crop = frame[y1:y2, x1:x2]
                    
                    if cow_crop.size > 0:
                        # Classify breed
                        breed, breed_conf = self.classify_breed(cow_crop)
                        
                        results.append({
                            'bbox': (x1, y1, x2, y2),
                            'detection_conf': confidence,
                            'breed': breed,
                            'breed_conf': breed_conf
                        })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            breed = result['breed']
            breed_conf = result['breed_conf']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{breed}: {breed_conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame