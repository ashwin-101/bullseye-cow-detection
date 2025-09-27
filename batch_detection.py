import cv2
import os
import argparse
from cow_detector import CowDetectorClassifier

def process_images(detector, input_dir, output_dir):
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            frame = cv2.imread(img_path)
            
            if frame is not None:
                # Process frame
                results = detector.process_frame(frame)
                
                # Draw results
                frame = detector.draw_results(frame, results)
                
                # Save result
                output_path = os.path.join(output_dir, f"detected_{filename}")
                cv2.imwrite(output_path, frame)
                
                print(f"Processed: {filename} -> {len(results)} cows detected")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--breed_model', type=str, 
                       default='classifier.h5',
                       help='Path to breed classification model')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CowDetectorClassifier(args.breed_model, args.yolo_model)
    
    # Process images
    process_images(detector, args.input_dir, args.output_dir)
    
    print(f"Batch processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()