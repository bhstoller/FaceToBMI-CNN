# BMI prediction from images

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from mtcnn import MTCNN
import base64
from PIL import Image
from io import BytesIO

# Our custom VGG Face
from vgg_face import get_vgg_face_model, preprocess_input

# Image size
IMAGE_SIZE = (224, 224)

class BMIPredictor:
    
    def __init__(self, model_path=None):
        self.detector = MTCNN()
        self.model = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
    
    def create_model(self):
        # VGG Face model with the pre-trained weights
        base_model = get_vgg_face_model(include_top=False, input_shape=(224, 224, 3))
        
        # We are freezing the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Custom regression layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='linear')(x)  # BMI is a continuous value
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        return model
    
    def preprocess_face_image(self, image_path=None, img=None, method='mtcnn'):
        try:
            # Load the image
            if image_path:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    return None
                
                # Converting BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                print("No image provided")
                return None
            
            # Convert to RGB
            if len(img.shape) == 2: # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            if method == 'mtcnn':
                # MTCNN for face detection
                faces = self.detector.detect_faces(img)
                if not faces:
                    print(f"No face detected in the image")
                    # Use the entire image if there's no face detected
                    face_img = cv2.resize(img, IMAGE_SIZE)
                else:
                    # Getting the largest face
                    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = face['box']
                    
                    x, y = max(0, x), max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    # Cropping the face
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, IMAGE_SIZE)
            else:
                # Use the entire image if we don't use MTCNN
                face_img = cv2.resize(img, IMAGE_SIZE)
            
            # Preprocess the image for VGG Face
            face_img = preprocess_input(face_img, version=1)
            
            return face_img
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def preprocess_base64_image(self, base64_string):
        try:
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            img = np.array(img)
            
            face_img = self.preprocess_face_image(img=img, method='mtcnn')
            
            if face_img is None:
                return None
                
            return np.expand_dims(face_img, axis=0)
            
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return None
    
    def load_data(self, data_path, image_folder):
        df = pd.read_csv(data_path)
        
        # Arrays for features and labels
        X = []
        y = []
        
        total = len(df)
        for idx, row in df.iterrows():
            print(f"Processing image {idx+1}/{total}: {row['name']}")
            
            image_path = os.path.join(image_folder, row['name'])
            
            # Checking if theres and image
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found")
                continue
            
            img = self.preprocess_face_image(image_path=image_path)
            if img is not None:
                X.append(img)
                y.append(row['bmi'])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=50, batch_size=4, validation_split=0.2, callbacks=None):
        if self.model is None:
            self.create_model()
            
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=callbacks
        )
        
        return history
    
    def predict_bmi(self, image_path=None, img=None, base64_string=None):
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Processing the image
        if base64_string:
            processed_image = self.preprocess_base64_image(base64_string)
        elif image_path or img is not None:
            face_img = self.preprocess_face_image(image_path=image_path, img=img)
            if face_img is None:
                return None, None
            processed_image = np.expand_dims(face_img, axis=0)
        else:
            raise ValueError("No image provided. Provide image_path.")
        
        if processed_image is None:
            return None, None
            
        # Making the prediction
        predicted_bmi = float(self.model.predict(processed_image)[0][0])
        
        # BMI category
        category = 'Unknown'
        if predicted_bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= predicted_bmi < 25:
            category = 'Normal weight'
        elif 25 <= predicted_bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obesity'
        
        return predicted_bmi, category
    
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("No model to save.")
            
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_plot(self, history, filename='training_history.png'):
        plt.figure(figsize=(12, 4))
        
        # Training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Training & validation mean absolute error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()