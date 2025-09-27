import cv2
import argparse
from cow_detector import CowDetectorClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--breed_model', type=str, 
                       default='classifier.h5',
                       help='Path to breed classification model')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CowDetectorClassifier(args.breed_model, args.yolo_model)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Draw results
        frame = detector.draw_results(frame, results)
        
        # Display frame
        cv2.imshow('Cow Detection & Breed Classification', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()