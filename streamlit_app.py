import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Import our BMI predictor - only when needed
from bmi_prediction import BMIPredictor

# Set up paths (changed to relative paths)
MODEL_PATH = "bmi_model.h5"

# Page configuration
st.set_page_config(
    page_title="Face to BMI Prediction",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Function to display BMI category with color
def display_bmi_category(bmi, category):
    if category == 'Underweight':
        st.markdown(f"<div style='background-color:#89CFF0;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Normal weight':
        st.markdown(f"<div style='background-color:#90EE90;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Overweight':
        st.markdown(f"<div style='background-color:#FFD700;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)
    elif category == 'Obesity':
        st.markdown(f"<div style='background-color:#FFA07A;padding:10px;border-radius:5px;text-align:center;'><strong>Category:</strong> {category} (BMI: {bmi:.2f})</div>", unsafe_allow_html=True)

# Function to preprocess and predict BMI from an image
def predict_from_image(image):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return
    
    # Only import predictor when needed
    predictor = BMIPredictor(MODEL_PATH)
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Process face
    with st.spinner("Processing image..."):
        try:
            # Detect face
            faces = predictor.detector.detect_faces(img_array)
            
            if not faces:
                st.warning("No face detected in the image. Using the entire image.")
                processed_img = cv2.resize(img_array, (224, 224))
                
                # Display the processed image
                st.image(processed_img, caption="Processed Image (No face detected)", use_column_width=True)
            else:
                # Get the largest face
                face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                x, y, w, h = face['box']
                
                # Ensure coordinates are valid
                x, y = max(0, x), max(0, y)
                w = min(w, img_array.shape[1] - x)
                h = min(h, img_array.shape[0] - y)
                
                # Draw rectangle on the face
                img_with_rect = img_array.copy()
                cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display the image with rectangle
                st.image(img_with_rect, caption="Detected Face", use_column_width=True)
                
                # Crop and process the face
                face_img = img_array[y:y+h, x:x+w]
                processed_img = cv2.resize(face_img, (224, 224))
            
            # Make prediction
            bmi, category = predictor.predict_bmi(img=processed_img)
            
            if bmi is None:
                st.error("Could not predict BMI from the image.")
                return
            
            # Display results
            st.subheader("Prediction Results")
            display_bmi_category(bmi, category)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("For developers: Check the console for more details")
            import traceback
            traceback.print_exc()
            return

# Main app
st.title("Face to BMI Prediction")
st.write("Upload a face image to predict BMI")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. The app needs a trained model to work.")
else:
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Display the original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image and predict
            predict_from_image(image)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()

# Footer
st.markdown("---")
st.caption("Face to BMI Prediction App - Based on VGG-Face and neural networks")