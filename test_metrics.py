# BMI Prediction Evaluation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# BMI predictor
from bmi_prediction import BMIPredictor

DATA_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/data.csv"
IMAGE_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/Images"
MODEL_PATH = "/Users/casey/Documents/GitHub/Face_BMI/bmi_model.h5"
RESULTS_PATH = "/Users/casey/Documents/GitHub/Face_to_BMI/evaluation_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for regression
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R^2': r2
    }

def plot_metrics(y_true, y_pred, metrics, save_path):
    """
    Create and save visualization plots for model evaluation
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        metrics: Dictionary of metric values
        save_path: Directory to save plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_xlabel('Actual BMI')
    ax1.set_ylabel('Predicted BMI')
    ax1.set_title('Actual vs Predicted BMI')
    
    # Add metrics annotation
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    # 2. Residuals plot
    ax2 = fig.add_subplot(2, 2, 2)
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted BMI')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    
    # 3. Histogram of residuals
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(residuals, bins=20, alpha=0.5, color='blue')
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Histogram of Residuals')
    
    # 4. Error distribution by BMI category
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Create BMI categories
    categories = []
    for bmi in y_true:
        if bmi < 18.5:
            categories.append('Underweight')
        elif 18.5 <= bmi < 25:
            categories.append('Normal weight')
        elif 25 <= bmi < 30:
            categories.append('Overweight')
        else:
            categories.append('Obesity')
    
    # Group errors by category
    unique_categories = sorted(set(categories))
    category_errors = {cat: [] for cat in unique_categories}
    
    for i, cat in enumerate(categories):
        abs_error = abs(y_true[i] - y_pred[i])
        category_errors[cat].append(abs_error)
    
    # Plot boxplot of errors by category
    box_data = [category_errors[cat] for cat in unique_categories]
    ax4.boxplot(box_data, labels=unique_categories)
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Error Distribution by BMI Category')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'bmi_prediction_metrics.png'))
    plt.close()

def main():
    print("=== BMI Prediction Model Evaluation ===")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return
    
    # Load the predictor with the trained model
    print("Loading trained model...")
    predictor = BMIPredictor(MODEL_PATH)
    
    # Load the dataset
    print("Loading and preprocessing dataset...")
    X, y = predictor.load_data(DATA_PATH, IMAGE_PATH)
    
    if len(X) == 0:
        print("Error: No valid images were loaded. Check your data paths.")
        return
    
    print(f"Successfully loaded {len(X)} images with corresponding BMI values")
    
    # Split dataset into training and test sets (or use all data if already tested on separate data)
    print("Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions on test data
    print(f"Making predictions on {len(X_test)} test samples...")
    y_pred = []
    
    for i, img in enumerate(X_test):
        # Reshape for single prediction
        img_reshaped = np.expand_dims(img, axis=0)
        pred = float(predictor.model.predict(img_reshaped)[0][0])
        y_pred.append(pred)
        
        # Print progress
        if (i+1) % 10 == 0 or i+1 == len(X_test):
            print(f"Processed {i+1}/{len(X_test)} images")
    
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    
    # Display metrics
    print("\n=== Evaluation Metrics ===")
    max_key_len = max(len(k) for k in metrics.keys())
    for key, value in metrics.items():
        print(f"{key.ljust(max_key_len)} : {value:.4f}")
    
    # Generate and save plots
    print("Generating evaluation plots...")
    plot_metrics(y_test, y_pred, metrics, RESULTS_PATH)
    
    # Save predictions to CSV for further analysis
    results_df = pd.DataFrame({
        'Actual_BMI': y_test,
        'Predicted_BMI': y_pred,
        'Absolute_Error': np.abs(y_test - y_pred),
        'Percentage_Error': np.abs((y_test - y_pred) / y_test) * 100
    })
    
    # Add BMI category columns
    results_df['Actual_Category'] = results_df['Actual_BMI'].apply(lambda x: 
        'Underweight' if x < 18.5 else
        'Normal weight' if 18.5 <= x < 25 else
        'Overweight' if 25 <= x < 30 else
        'Obesity'
    )
    
    results_df['Predicted_Category'] = results_df['Predicted_BMI'].apply(lambda x: 
        'Underweight' if x < 18.5 else
        'Normal weight' if 18.5 <= x < 25 else
        'Overweight' if 25 <= x < 30 else
        'Obesity'
    )
    
    # Add category match column
    results_df['Category_Match'] = results_df['Actual_Category'] == results_df['Predicted_Category']
    
    csv_path = os.path.join(RESULTS_PATH, 'bmi_prediction_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate category accuracy
    category_accuracy = results_df['Category_Match'].mean() * 100
    print(f"\nBMI Category Prediction Accuracy: {category_accuracy:.2f}%")
    
    # Analyze errors by BMI category
    print("\n=== Error Analysis by BMI Category ===")
    category_stats = results_df.groupby('Actual_Category').agg({
        'Absolute_Error': ['mean', 'std', 'min', 'max'],
        'Percentage_Error': ['mean', 'std'],
        'Category_Match': 'mean'
    })
    
    print(category_stats)
    
    # Save detailed statistics
    stats_path = os.path.join(RESULTS_PATH, 'bmi_category_stats.csv')
    category_stats.to_csv(stats_path)
    print(f"Category statistics saved to {stats_path}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()

# Evaluation Metrics
#MSE  : 209.2923
#RMSE : 14.4669
#MAE  : 12.8457
#MAPE : 37.7889
#R^2  : -2.2726

#BMI Category Prediction Accuracy: 6.18%
# Error Analysis by BMI Category
#                Absolute_Error                                Percentage_Error            Category_Match
#                       mean       std       min        max             mean        std           mean
#Actual_Category                                                                                         
#Normal weight         6.242294  2.714545  0.161071  12.611897        26.845618  11.207775       0.267123
#Obesity              16.922451  6.092076  3.103079  42.441495        43.773536   9.490337       0.016279
#Overweight            9.210173  2.840872  1.152081  17.563313        33.292608   9.583244       0.013825