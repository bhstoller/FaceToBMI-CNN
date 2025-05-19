# test_save.py
import os
import sys
from bmi_prediction import BMIPredictor

def test_model_save():
    try:
        print("Creating BMI predictor...")
        predictor = BMIPredictor()
        
        print("Creating model...")
        model = predictor.create_model()
        
        # Print model summary
        print("Model summary:")
        model.summary()
        
        # Try to save model
        test_path = "test_model.h5"
        print(f"Attempting to save model to {test_path}...")
        predictor.save_model(test_path)
        
        # Verify
        if os.path.exists(test_path):
            file_size_mb = os.path.getsize(test_path) / (1024 * 1024)
            print(f"SUCCESS: Model saved to {test_path}")
            print(f"File size: {file_size_mb:.2f} MB")
            return True
        else:
            print(f"FAILED: File not found at {test_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_save()
    sys.exit(0 if success else 1)

# Add this to your test_save.py

def test_prediction():
    try:
        model_path = "test_model.h5"
        if not os.path.exists(model_path):
            print("No model found. Run test_model_save() first.")
            return False
            
        predictor = BMIPredictor(model_path)
        
        # Test image path
        test_image_path = "test_face.jpg" 
        if not os.path.exists(test_image_path):
            print(f"Test image not found at {test_image_path}")
            return False
            
        # Make prediction
        print(f"Testing prediction with image: {test_image_path}")
        bmi, category = predictor.predict_bmi(image_path=test_image_path)
        
        if bmi is None:
            print("Prediction failed.")
            return False
            
        print(f"Prediction successful!")
        print(f"Predicted BMI: {bmi:.2f}")
        print(f"Category: {category}")
        return True
        
    except Exception as e:
        print(f"ERROR during prediction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Add this to the main section
if __name__ == "__main__":
    success = test_model_save()
    if success:
        # Try a prediction
        test_prediction()
    sys.exit(0 if success else 1)