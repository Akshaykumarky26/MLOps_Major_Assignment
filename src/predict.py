import joblib
import os
from src.utils import load_data # Re-using existing utility for consistency
import numpy as np # Needed for reshaping input for prediction

def main():
    # Load trained model 
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    # Simple check for model existence, crucial for Docker context
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Ensure it's copied into the Docker image.")
        exit(1) # Exit if the model isn't found

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Run prediction on test set 
    _, X_test, _, y_test = load_data()
    y_pred = model.predict(X_test)
    print("Predictions made on the test set.")

    # Print sample outputs 
    print("\n--- Sample Outputs ---")
    num_samples = 3 # Printing first 3 samples
    for i in range(min(num_samples, len(X_test))):
        print(f"Sample {i+1}:")
        print(f"  Actual: {y_test[i]:.4f}")
        print(f"  Predicted: {y_pred[i]:.4f}")
    print("--------------------")

if __name__ == "__main__":
    main()