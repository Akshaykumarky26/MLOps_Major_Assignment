import joblib
import numpy as np
import os
from src.utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def quantize_model(model_path='models/linear_regression_model.joblib'):
    print(f"Loading trained model from {model_path} for quantization...")
    model = joblib.load(model_path)

    raw_coef = model.coef_
    raw_intercept = model.intercept_

    os.makedirs('parameters', exist_ok=True)

    # Save unquantized parameters
    joblib.dump({'coef': raw_coef, 'intercept': raw_intercept}, 'parameters/unquant_params.joblib')
    print("Raw parameters saved to parameters/unquant_params.joblib")

    # === Robust Global Scale Quantization ===
    clipped_coef = np.copy(raw_coef)
    clipped_coef[np.abs(clipped_coef) < 1e-4] = 0  # Avoid exploding scale
    max_abs_coef = np.max(np.abs(clipped_coef))

    if max_abs_coef == 0:
        scale = 1.0
        quant_coef = np.full_like(raw_coef, 128, dtype=np.uint8)
    else:
        scale = 255.0 / (2 * max_abs_coef)
        zero_point = 128.0
        scaled = clipped_coef * scale + zero_point
        quant_coef = np.clip(np.round(scaled), 0, 255).astype(np.uint8)

    joblib.dump({
        'quant_coef': quant_coef,
        'scale': scale,
        'zero_point': 128.0,
        'raw_intercept': float(raw_intercept)
    }, 'parameters/quant_params.joblib')
    print("Quantized parameters saved to parameters/quant_params.joblib")

    # === Dequantization ===
    dequant_coef = (quant_coef.astype(np.float32) - 128.0) / scale
    dequant_intercept = float(raw_intercept)

    dequant_model = LinearRegression()
    dequant_model.coef_ = dequant_coef
    dequant_model.intercept_ = dequant_intercept

    _, X_test, _, y_test = load_data()
    y_pred = dequant_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"De-quantized Model R^2 Score: {r2:.4f}")
    print(f"De-quantized Model Mean Squared Error: {mse:.4f}")

    print(f"\nSample actual value: {y_test[0]:.4f}")
    print(f"Sample original model prediction: {model.predict(X_test[0].reshape(1, -1))[0]:.4f}")
    print(f"Sample de-quantized model prediction: {y_pred[0]:.4f}")

if __name__ == "__main__":
    quantize_model()
