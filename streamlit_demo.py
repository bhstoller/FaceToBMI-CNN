import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import random
import time
import base64
from io import BytesIO
import os

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Face to BMI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Face to BMI Prediction\nAn AI application that predicts BMI from facial features."
    }
)

# Function to get image as base64 string
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background from local image
def set_background(image_file):
    try:
        bin_str = get_base64_of_bin_file(image_file)
        page_bg_img = '''
        <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        pass

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
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
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        color: #7f8c8d;
        font-size: 0.8em;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load default brain icon
def get_brain_icon():
    return "https://www.svgrepo.com/show/13656/brain.svg"

# Sidebar content
st.sidebar.image(get_brain_icon(), width=100)
st.sidebar.title("Face to BMI")
st.sidebar.markdown("---")

# Add sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Demo", "About", "Research"])

# Function to create fancy face detection effect
def process_image_with_effects(image):
    # Create a copy and apply some enhancements
    enhanced = image.copy()
    
    # Slightly enhance contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.2)
    
    # Slightly enhance color
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    # Get dimensions
    width, height = enhanced.size
    
    # Create a drawing surface
    draw = ImageDraw.Draw(enhanced)
    
    # Calculate face position (simplified - center of image)
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 2
    x = center_x - box_size // 2
    y = center_y - box_size // 2
    
    # Draw a fancy border
    border_width = 3
    draw.rectangle(
        [(x, y), (x + box_size, y + box_size)], 
        outline=(41, 128, 185), 
        width=border_width
    )
    
    # Draw corner markers
    corner_size = 15
    # Top left
    draw.line([(x, y), (x + corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y), (x, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Top right
    draw.line([(x + box_size, y), (x + box_size - corner_size, y)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + box_size, y), (x + box_size, y + corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom left
    draw.line([(x, y + box_size), (x + corner_size, y + box_size)], fill=(231, 76, 60), width=border_width)
    draw.line([(x, y + box_size), (x, y + box_size - corner_size)], fill=(231, 76, 60), width=border_width)
    # Bottom right
    draw.line([(x + box_size, y + box_size), (x + box_size - corner_size, y + box_size)], fill=(231, 76, 60), width=border_width)
    draw.line([(x + box_size, y + box_size), (x + box_size, y + box_size - corner_size)], fill=(231, 76, 60), width=border_width)
    
    # Add some scan lines for a "processing" effect
    for i in range(y, y + box_size, 10):
        draw.line([(x, i), (x + box_size, i)], fill=(46, 204, 113, 128), width=1)
    
    return enhanced

# Function to display BMI category with beautiful styling
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
    cols = st.columns([1, 1, 1, 1])
    
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

# Simulated face detection and BMI prediction
def demo_prediction(image):
    with st.spinner("Processing image..."):
        # Simulate AI thinking
        thinking_placeholder = st.empty()
        thinking_steps = [
            "Detecting face...",
            "Analyzing facial features...",
            "Extracting measurements...",
            "Applying BMI prediction model...",
            "Finalizing results..."
        ]
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for i, step in enumerate(thinking_steps):
            thinking_placeholder.markdown(f"<div style='color:#3498db;'>{step}</div>", unsafe_allow_html=True)
            for j in range(20):
                progress = (i * 20 + j) / (len(thinking_steps) * 20)
                progress_bar.progress(progress)
                time.sleep(0.03)
        
        progress_bar.progress(100)
        thinking_placeholder.empty()
        
        # Process image with fancy effects
        processed_image = process_image_with_effects(image)
        
        # Display the processed image
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(processed_image, caption="AI Face Detection", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate a random but realistic BMI
        # Slightly skew toward normal range (18.5-25)
        bmi_base = random.normalvariate(23, 4)  # Mean of 23, std dev of 4
        bmi = max(15, min(40, bmi_base))  # Clamp between 15 and 40
        
        # Determine BMI category
        if bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= bmi < 25:
            category = 'Normal weight'
        elif 25 <= bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obesity'
        
        # Simulate final calculation with countdown
        st.markdown("<h3 style='text-align:center;'>Calculating Final Result</h3>", unsafe_allow_html=True)
        result_placeholder = st.empty()
        for i in range(3, 0, -1):
            result_placeholder.markdown(f"<h1 style='text-align:center;'>{i}</h1>", unsafe_allow_html=True)
            time.sleep(0.5)
        result_placeholder.empty()
        
        # Display results section
        st.markdown("<h2 style='text-align:center; margin-top:30px; margin-bottom:30px;'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # Add notification that this is a demo
        st.markdown("""
        <div style="background-color:#f8d7da; border-left:5px solid #dc3545; padding:15px; margin-bottom:20px; border-radius:4px;">
            <strong>Demo Mode:</strong> This is a simulated prediction for demonstration purposes only. 
            In the full app, a real AI model would make predictions based on actual facial features.
        </div>
        """, unsafe_allow_html=True)
        
        # Display the BMI results
        display_bmi_category(bmi, category)
        
        # Add a fun fact about BMI
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

# Page content based on selection
if page == "Home":
    # Homepage
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Face to BMI Prediction")
        st.markdown("""
        <div class="info-box">
            Welcome to the Face to BMI Prediction app! This cutting-edge tool demonstrates how artificial intelligence can estimate Body Mass Index (BMI) from facial features.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### How It Works")
        st.write("""
        The full application uses a deep learning model built on the VGG-Face architecture to analyze facial features and predict BMI. The model is trained on a dataset of face images with known BMI values.
        
        This demo version illustrates the user interface and shows how the predictions would be presented.
        """)
        
        with st.expander("See the Technology Stack"):
            st.write("""
            - **AI Model**: Transfer learning with VGG-Face
            - **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
            - **Backend**: TensorFlow/Keras
            - **Frontend**: Streamlit
            - **Image Processing**: OpenCV, PIL
            """)
    
    with col2:
        st.image(get_brain_icon(), width=200)
        st.markdown("""
        <div style="background-color:#e8f8f5; padding:15px; border-radius:4px; text-align:center; margin-top:20px;">
            <h3 style="margin-top:0;">Try the Demo</h3>
            <p>Upload your photo and see a simulated BMI prediction!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Demo", key="home_demo_button"):
            st.session_state.page = "Demo"
            st.experimental_rerun()

elif page == "Demo":
    # Demo page
    st.title("BMI Prediction Demo")
    st.markdown("""
    <div class="info-box">
        Upload a face image below to see how the BMI prediction would work.
        This demo uses simulated predictions for illustration purposes.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
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
            predict_button = st.button("Generate Prediction", key="predict_button")
    
    with col2:
        if uploaded_file is not None and predict_button:
            # Process the image and show prediction
            demo_prediction(image)
        else:
            # Show placeholder content
            st.markdown("### Prediction Results")
            st.markdown("""
            <div style="background-color:#f5f5f5; padding:30px; border-radius:8px; text-align:center; margin-top:20px; color:#777;">
                <img src="https://www.svgrepo.com/show/13656/brain.svg" width="80" style="margin-bottom:15px;">
                <h3>Upload an image and click 'Generate Prediction'</h3>
                <p>The results will appear here</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample images
    st.markdown("---")
    st.markdown("### Don't have an image? Try one of these examples:")
    
    sample_cols = st.columns(4)
    
    # These would normally be local files, but for the demo we'll use placeholders
    sample_images = [
        "https://preview.redd.it/person-with-straight-face-staring-directly-into-the-v0-wgnbxn8r0z3c1.jpg?width=150&format=pjpg&auto=webp&s=7c1cb68c1bac3f52e25e14d7fb622ce6ea3d9c23",
        "https://cdn.optinmonster.com/wp-content/uploads/2022/04/Ultimate-Guide-to-Persona-Marketing-1-150x150.png",
        "https://images.squarespace-cdn.com/content/v1/57e74a5fd2b857ab5fcd5ad9/1600302535236-GQ2URM3KHZRH7IXIUCDM/professional+headshot+on+simple+white+background.jpg?format=150w",
        "https://media.licdn.com/dms/image/C5103AQHQPfTnDK7wqQ/profile-displayphoto-shrink_200_200/0/1522845139132?e=1720742400&v=beta&t=52zVPzXS2e51vQz7dz_8G3oc2ZoGH7AbIWyPKXxHx8s"
    ]
    
    for i, (col, img_url) in enumerate(zip(sample_cols, sample_images)):
        with col:
            st.image(img_url, width=120)
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                # In a real app, you would load a local file here
                st.warning("In the full app, this would load a sample image from your files.")

elif page == "About":
    # About page
    st.title("About Face to BMI Prediction")
    
    st.markdown("""
    <div class="info-box">
        This application demonstrates the potential of AI in analyzing facial features to predict physical attributes.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    ### Project Overview
    
    Face to BMI is a research-based application that explores the correlation between facial features and Body Mass Index (BMI). Using deep learning and computer vision techniques, the system analyzes facial characteristics to make BMI predictions.
    
    ### Key Features
    
    - **Non-invasive BMI Estimation**: Predicts BMI without physical measurements
    - **Advanced AI**: Leverages transfer learning with pre-trained facial recognition models
    - **User-friendly Interface**: Simple upload-and-predict workflow
    
    ### Technical Implementation
    
    The full system uses a convolutional neural network based on the VGG-Face architecture. The pre-trained model is fine-tuned on a dataset of facial images with known BMI values, creating a regression model that outputs BMI predictions.
    """)
    
    # Team information
    st.markdown("### Project Team")
    
    # Team members - updated with the provided names
    team_members = [
        {"name": "Kyler Rose", "role": "Team Member"},
        {"name": "Cassandra Maldonado", "role": "Team Member"},
        {"name": "Bradley Stoller", "role": "Team Member"},
        {"name": "Bruna Medeiros", "role": "Team Member"},
        {"name": "Yu-Hsuan Ko", "role": "Team Member"},
        {"name": "Angus Ho", "role": "Team Member"}
    ]
    
    # Display team members in rows of 3
    for i in range(0, len(team_members), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(team_members):
                with cols[j]:
                    st.markdown(f"""
                    <div style="text-align:center; padding:15px; border-radius:8px; background-color:#f5f7f9;">
                        <img src="https://www.svgrepo.com/show/335197/avatar.svg" width="100">
                        <h3>{team_members[i+j]['name']}</h3>
                        <p>{team_members[i+j]['role']}</p>
                    </div>
                    """, unsafe_allow_html=True)

elif page == "Research":
    # Research page
    st.title("Research Background")
    
    st.markdown("""
    <div class="info-box">
        This project is based on scientific research exploring the relationship between facial morphology and body mass index.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    ### Scientific Foundation
    
    Recent studies have demonstrated correlations between facial features and BMI. Facial adiposity (fat storage) changes visibly with overall body composition, creating patterns that can be detected by sophisticated computer vision systems.
    
    The research paper "Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media" provides the theoretical foundation for this application.
    
    ### How It Works
    
    1. **Face Detection**: The system first identifies and isolates the face in an image
    2. **Feature Extraction**: Deep convolutional layers extract complex facial features
    3. **Regression Analysis**: These features are analyzed to predict a continuous BMI value
    
    ### Limitations
    
    It's important to acknowledge the limitations of this approach:
    
    - BMI itself has limitations as a health metric
    - Predictions are estimates and not substitutes for medical measurements
    - Genetic, ethnic, and age factors may influence accuracy
    - The system is a research demonstration, not a medical diagnostic tool
    """)
    
    # Add a research visualization
    st.markdown("### Conceptual Framework")
    
    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*XtCMwEXZe2VcH-jfCQGl1A.jpeg", 
             caption="Neural network analyzing facial features (Conceptual illustration)", 
             use_column_width=True)
    
    # Updated Research Citation
    st.markdown("### Research Citation")
    st.markdown("""
    Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media  
    Enes Kocabey, Mustafa Camurcu, Ferda Ofli, Yusuf Aytar, Javier Marin, Antonio Torralba, Ingmar Weber
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Face to BMI Prediction | Research Project</p>
    <p>This is a demo application for educational and research purposes. Not intended for medical use.</p>
</div>
""", unsafe_allow_html=True)