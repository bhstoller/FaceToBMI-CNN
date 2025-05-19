import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import random
import time

# Set page configuration
st.set_page_config(
    page_title="Face to BMI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f1f8ff;
        border-left: 5px solid #4b6fff;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .img-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .category-badge {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Class to simulate BMI prediction (without external dependencies)
class SimulatedBMIPredictor:
    def detect_face(self, image):
        """Simulate face detection"""
        # Get image dimensions
        width, height = image.size
        
        # Create a simulated face box in the center of the image
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x = center_x - box_size // 2
        y = center_y - box_size // 2
        
        return (x, y, box_size, box_size)
    
    def predict_bmi(self, image):
        """Simulate BMI prediction"""
        # In a real model, this would analyze facial features
        # Here we'll use a consistent random value based on image properties
        
        # Get image dimensions and calculate a seed value
        width, height = image.size
        seed_value = (width * height) % 1000
        random.seed(seed_value)
        
        # Generate a BMI value (slightly skewed towards normal range)
        bmi_base = random.normalvariate(23, 4)
        bmi = max(15, min(40, bmi_base))
        
        # Determine BMI category
        if bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= bmi < 25:
            category = 'Normal weight'
        elif 25 <= bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obesity'
        
        return bmi, category

# Function to display BMI category with styling
def display_bmi_category(bmi, category):
    if category == 'Underweight':
        color = "#3498db"
        icon = "⬇️"
        description = "BMI less than 18.5 indicates underweight. This may be associated with certain health issues."
    elif category == 'Normal weight':
        color = "#2ecc71"
        icon = "✅"
        description = "BMI between 18.5 and 24.9 indicates a healthy weight for most adults."
    elif category == 'Overweight':
        color = "#f39c12"
        icon = "⚠️"
        description = "BMI between 25 and 29.9 indicates overweight. This may increase risk for certain diseases."
    elif category == 'Obesity':
        color = "#e74c3c"
        icon = "❗"
        description = "BMI of 30 or higher indicates obesity. This increases risk for many health conditions."
    
    st.markdown(f"""
    <div class="category-badge" style="background-color:{color}20; border-left:5px solid {color}; color:{color};">
        <h2 style="margin:0;">{icon} {category} (BMI: {bmi:.1f})</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom:25px;">
        {description}
    </div>
    """, unsafe_allow_html=True)
    
    # Add a visualization gauge
    st.markdown("### BMI Scale")
    
    # Calculate position on the BMI scale (0-100%)
    if bmi < 15:
        position = 0
    elif bmi > 35:
        position = 100
    else:
        position = (bmi - 15) * 100 / 20  # Scale 15-35 to 0-100%
    
    # Create the gauge sections
    gauge_html = f"""
    <div style="display:flex; height:25px; border-radius:5px; overflow:hidden; margin-bottom:10px; width:100%;">
        <div style="background-color:#3498db; width:17.5%; height:100%;"></div>
        <div style="background-color:#2ecc71; width:32.5%; height:100%;"></div>
        <div style="background-color:#f39c12; width:25%; height:100%;"></div>
        <div style="background-color:#e74c3c; width:25%; height:100%;"></div>
    </div>
    <div style="position:relative; height:20px; margin-bottom:20px;">
        <div style="position:absolute; left:0%;">15</div>
        <div style="position:absolute; left:17.5%;">18.5</div>
        <div style="position:absolute; left:50%;">25</div>
        <div style="position:absolute; left:75%;">30</div>
        <div style="position:absolute; right:0%;">35+</div>
        <div style="position:absolute; left:{position}%; transform:translateX(-50%); top:-25px;">
            <div style="width:0; height:0; border-left:10px solid transparent; border-right:10px solid transparent; border-top:10px solid #333; margin:0 auto;"></div>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

# Function to highlight detected face
def draw_face_detection(image, face_box):
    # Create a copy of the image
    enhanced = image.copy()
    
    # Create a drawing surface
    draw = ImageDraw.Draw(enhanced)
    
    # Extract coordinates
    x, y, w, h = face_box
    
    # Draw a fancy border
    border_width = 3
    draw.rectangle(
        [(x, y), (x + w, y + h)], 
        outline=(41, 128, 185), 
        width=border_width
    )
    
    # Draw corner markers
    corner_size = 15
    # Top left
    draw.line([(x, y), (x + corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y), (x, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Top right
    draw.line([(x + w, y), (x + w - corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + w, y), (x + w, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom left
    draw.line([(x, y + h), (x + corner_size, y + h)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y + h), (x, y + h - corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom right
    draw.line([(x + w, y + h), (x + w - corner_size, y + h)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + w, y + h), (x + w, y + h - corner_size)], fill=(231, 76, 60), width=border_width)
    
    # Add some scan lines for a "processing" effect
    for i in range(y, y + h, 10):
        draw.line([(x, i), (x + w, i)], fill=(46, 204, 113, 128), width=1)
    
    return enhanced

# Function to make BMI prediction
def predict_bmi(image):
    with st.spinner("Processing image..."):
        # Initialize our simulated predictor
        predictor = SimulatedBMIPredictor()
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize
        status_text.text("Initializing BMI predictor...")
        time.sleep(0.5)
        progress_bar.progress(20)
        
        # Step 2: Detect face
        status_text.text("Detecting face in image...")
        time.sleep(0.8)
        
        # Simulated face detection
        face_box = predictor.detect_face(image)
        
        # Draw detection on image
        enhanced_image = draw_face_detection(image, face_box)
        
        # Display the processed image
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(enhanced_image, caption="Face Detection", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        progress_bar.progress(60)
        
        # Step 3: Predict BMI
        status_text.text("Analyzing facial features and predicting BMI...")
        time.sleep(1.2)
        
        # Simulated BMI prediction
        bmi, category = predictor.predict_bmi(image)
        progress_bar.progress(80)
        
        # Step 4: Display results
        status_text.text("Preparing results...")
        time.sleep(0.5)  # Brief pause for visual effect
        progress_bar.progress(100)
        status_text.empty()
        
        # Display results header
        st.markdown("<h2 style='text-align:center; margin-top:30px; margin-bottom:30px;'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # Add demo notification
        st.markdown("""
        <div style="background-color:#f8d7da; border-left:5px solid #dc3545; padding:15px; margin-bottom:20px; border-radius:4px;">
            <strong>Demo Mode:</strong> This is a simulated prediction for demonstration purposes.
            The real app would use a trained machine learning model to analyze facial features.
        </div>
        """, unsafe_allow_html=True)
        
        # Display the BMI results
        display_bmi_category(bmi, category)
        
        # Add a BMI fact
        bmi_facts = [
            "BMI was developed by Belgian mathematician Adolphe Quetelet in the 1830s.",
            "BMI doesn't distinguish between weight from muscle and weight from fat.",
            "Athletes often have a high BMI due to muscle mass, despite being very fit.",
            "BMI categories may vary slightly for different ethnic groups.",
            "The BMI formula is weight (kg) divided by height squared (m²)."
        ]
        
        st.markdown(f"""
        <div style="background-color:#e8f4f8; padding:15px; border-radius:4px; margin-top:30px;">
            <strong>BMI Fact:</strong> {random.choice(bmi_facts)}
        </div>
        """, unsafe_allow_html=True)

# Main app
st.title("Face to BMI Prediction")
st.markdown("""
<div class="info-box">
    Welcome to the Face to BMI Prediction app! This demo simulates how AI can analyze facial features to estimate Body Mass Index (BMI).
</div>
""", unsafe_allow_html=True)

# Create two columns for the interface
col1, col2 = st.columns([1, 1])

with col1:
    # Upload image
    st.markdown("### Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Display the original image
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Button to trigger prediction
        predict_button = st.button("Predict BMI", key="predict_button")

with col2:
    if uploaded_file is not None and predict_button:
        # Process the image and show prediction
        predict_bmi(image)
    else:
        # Show instructions
        st.markdown("### How It Works")
        st.markdown("""
        This demonstration app shows how facial features could be analyzed to predict BMI:
        
        1. **Upload an image** with a clearly visible face
        2. Click **"Predict BMI"**
        3. The app will:
           - Detect the face in the image
           - Analyze facial features
           - Estimate BMI based on these features
           - Display results with category and visualization
        
        In the full application, this would use a deep learning model based on VGG-Face architecture trained on a dataset of facial images with known BMI values.
        """)
        
        st.markdown("""
        <div style="background-color:#e8f8f5; padding:15px; border-radius:4px; margin-top:20px;">
            <h4 style="margin-top:0;">Research Citation</h4>
            <p>Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media<br>
            Enes Kocabey, Mustafa Camurcu, Ferda Ofli, Yusuf Aytar, Javier Marin, Antonio Torralba, Ingmar Weber</p>
        </div>
        """, unsafe_allow_html=True)

# Team section
st.markdown("---")
st.markdown("### Project Team")

# Team members - showing in two rows
team_members = [
    "Kyler Rose", "Cassandra Maldonado", "Bradley Stoller", 
    "Bruna Medeiros", "Yu-Hsuan Ko", "Angus Ho"
]

# First row
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, member in enumerate(team_members[:3]):
    with cols[i]:
        st.markdown(f"""
        <div style="text-align:center; padding:15px; border-radius:8px; background-color:#f5f7f9;">
            <img src="https://www.svgrepo.com/show/335197/avatar.svg" width="80">
            <h4 style="margin-bottom:5px;">{member}</h4>
            <p style="color:#666; margin:0;">Team Member</p>
        </div>
        """, unsafe_allow_html=True)

# Second row
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, member in enumerate(team_members[3:]):
    with cols[i]:
        st.markdown(f"""
        <div style="text-align:center; padding:15px; border-radius:8px; background-color:#f5f7f9;">
            <img src="https://www.svgrepo.com/show/335197/avatar.svg" width="80">
            <h4 style="margin-bottom:5px;">{member}</h4>
            <p style="color:#666; margin:0;">Team Member</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Face to BMI Prediction | Demo Version")