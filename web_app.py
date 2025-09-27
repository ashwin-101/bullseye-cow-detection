import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from cow_detector import CowDetectorClassifier

# Page config
st.set_page_config(
    page_title="Cow Detection & Breed Classification",
    page_icon="🐄",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None

def load_model():
    """Load the breed classification model"""
    model_path = "classifier.h5"
    if os.path.exists(model_path):
        try:
            st.session_state.detector = CowDetectorClassifier(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    else:
        st.error("Model file not found. Please check the path.")
        return False

def process_image(image):
    """Process a single image"""
    if st.session_state.detector is None:
        st.error("Please load the model first!")
        return None, []
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process frame
    results = st.session_state.detector.process_frame(img_array)
    
    # Draw results
    output_img = st.session_state.detector.draw_results(img_array.copy(), results)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    return output_img, results

def main():
    st.title("🐄 Cow Detection & Breed Classification")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Load model button
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            if load_model():
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model")
    
    # Model status
    if st.session_state.detector:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.warning("⚠️ Model Not Loaded")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "📁 Batch Processing"])
    
    with tab1:
        st.header("Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing cows"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                if st.button("Detect Cows", type="primary"):
                    with st.spinner("Processing..."):
                        output_img, results = process_image(image)
                        
                        if output_img is not None:
                            st.image(output_img, use_column_width=True)
                            
                            # Display results table
                            if results:
                                st.subheader("Detected Cows")
                                data = []
                                for i, result in enumerate(results, 1):
                                    data.append({
                                        "Cow #": i,
                                        "Breed": result['breed'],
                                        "Confidence": f"{result['breed_conf']:.2f}",
                                        "Detection Conf": f"{result['detection_conf']:.2f}"
                                    })
                                st.dataframe(data, use_container_width=True)
                            else:
                                st.info("No cows detected in the image")
    
    with tab2:
        st.header("Webcam Detection")
        st.info("Click 'Start' to begin webcam detection")
        
        # Webcam input
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            image = Image.open(camera_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                with st.spinner("Processing..."):
                    output_img, results = process_image(image)
                    
                    if output_img is not None:
                        st.image(output_img, use_column_width=True)
                        
                        if results:
                            st.subheader("Detected Cows")
                            for i, result in enumerate(results, 1):
                                st.write(f"**Cow {i}:** {result['breed']} ({result['breed_conf']:.2f})")
                        else:
                            st.info("No cows detected")
    
    with tab3:
        st.header("Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("Process All Images", type="primary"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                all_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    output_img, results = process_image(image)
                    
                    if output_img is not None:
                        all_results.append({
                            'filename': uploaded_file.name,
                            'image': output_img,
                            'results': results
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display all results
                with results_container:
                    for result in all_results:
                        st.subheader(f"Results for {result['filename']}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(result['image'], use_column_width=True)
                        
                        with col2:
                            if result['results']:
                                st.write("**Detected Cows:**")
                                for j, cow_result in enumerate(result['results'], 1):
                                    st.write(f"Cow {j}: {cow_result['breed']} ({cow_result['breed_conf']:.2f})")
                            else:
                                st.write("No cows detected")
                        
                        st.markdown("---")

if __name__ == "__main__":
    main()