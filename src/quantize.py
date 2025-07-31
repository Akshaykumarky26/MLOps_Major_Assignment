import joblib
import numpy as np
import os
from src.utils import load_data
from sklearn.linear_model import LinearRegression # Needed to reconstruct the model for de-quantization

def quantize_model(model_path='models/linear_regression_model.joblib'):
    print(f"Loading trained model from {model_path} for quantization...")
    model = joblib.load(model_path)

    # 1. Extract coef and intercept 
    raw_coef = model.coef_
    raw_intercept = model.intercept_

    # Create a directory to save parameters if it doesn't exist
    os.makedirs('parameters', exist_ok=True)

    # 2. Save raw parameters (unquant_params.joblib) 
    unquant_params_path = os.path.join('parameters', 'unquant_params.joblib')
    joblib.dump({'coef': raw_coef, 'intercept': raw_intercept}, unquant_params_path)
    print(f"Raw parameters saved to {unquant_params_path}")

    # 3. Manually quantize them to unsigned 8-bit integers 
    # Determine min and max values for scaling to [0, 255]
    all_params = np.concatenate((raw_coef.flatten(), [raw_intercept]))
    min_val = np.min(all_params)
    max_val = np.max(all_params)

    # Scale to [0, 255] and convert to uint8
    # Ensure division by zero is handled if min_val == max_val
    if max_val == min_val:
        scaled_coef = np.full_like(raw_coef, 127.5, dtype=np.float32) # Mid-point if all values are same
        scaled_intercept = 127.5
    else:
        scale = 255.0 / (max_val - min_val)
        scaled_coef = (raw_coef - min_val) * scale
        scaled_intercept = (raw_intercept - min_val) * scale

    quant_coef = np.round(scaled_coef).astype(np.uint8)
    quant_intercept = np.uint8(np.round(scaled_intercept))

    # Store min/max for de-quantization
    quant_metadata = {
        'min_val': min_val,
        'max_val': max_val,
        'quant_coef': quant_coef,
        'quant_intercept': quant_intercept
    }

    # 4. Save quantized parameters (quant_params.joblib) 
    quant_params_path = os.path.join('parameters', 'quant_params.joblib')
    joblib.dump(quant_metadata, quant_params_path)
    print(f"Quantized parameters saved to {quant_params_path}")

    # 5. Perform inference with the de-quantized weights 
    print("Performing inference with de-quantized weights...")
    # De-quantize the parameters back to float
    dequant_scale = (max_val - min_val) / 255.0
    dequant_coef = (quant_coef.astype(np.float32) * dequant_scale) + min_val
    dequant_intercept = (quant_intercept.astype(np.float32) * dequant_scale) + min_val

    # Create a new LinearRegression model with de-quantized parameters
    dequant_model = LinearRegression()
    dequant_model.coef_ = dequant_coef
    dequant_model.intercept_ = dequant_intercept

    # Ensuring that the model is recognized as fitted by scikit-learn
    dequant_model.n_features_in_ = model.n_features_in_ if hasattr(model, 'n_features_in_') else X_train.shape[1] # Use original model's features_in_ or infer

    # Load test data
    _, X_test, _, y_test = load_data()

    # Predict with de-quantized model
    y_pred_dequant = dequant_model.predict(X_test)

    # Compare a sample prediction
    print(f"Sample actual value: {y_test[0]:.4f}")
    print(f"Sample original model prediction: {model.predict(X_test[0].reshape(1, -1))[0]:.4f}")
    print(f"Sample de-quantized model prediction: {y_pred_dequant[0]:.4f}")

if __name__ == "__main__":
    quantize_model()