# tests/test_train.py
import os
import joblib
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data
from src.train import train_model

# Define a fixture to ensure the model is trained before tests run
@pytest.fixture(scope="module", autouse=True)
def setup_trained_model():
    """Ensures the model is trained and saved before any tests run."""
    # Run the training script to generate the model
    train_model()
    yield # This yields control to the tests
    # Teardown: Clean up the generated model file after tests are done (optional, but good practice)
    if os.path.exists('models/linear_regression_model.joblib'):
        os.remove('models/linear_regression_model.joblib')
        # Only remove the directory if it's empty. Use shutil.rmtree for non-empty dirs.
        # For simplicity, let's just remove the file for now, or use try-except for rmdir
        try:
            os.rmdir('models') # This will only succeed if the directory is empty
        except OSError as e:
            # If directory is not empty or access denied, print a message but don't fail the test session
            print(f"Warning: Could not remove 'models' directory: {e}")


def test_dataset_loading():
    """Unit test dataset loading."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_model_creation():
    """Validate model creation (LinearRegression instance)."""
    model_path = 'models/linear_regression_model.joblib'
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    model = joblib.load(model_path)
    assert isinstance(model, LinearRegression)

def test_model_trained():
    """Check if model was trained (e.g., coef_ exists)."""
    model_path = 'models/linear_regression_model.joblib'
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    model = joblib.load(model_path)
    # Check if coefficients exist, which implies the model has been fitted
    assert hasattr(model, 'coef_')
    assert model.coef_ is not None
    assert len(model.coef_) > 0 # Ensure coefficients are not empty

def test_r2_score_threshold():
    """Ensure R^2 score exceeds minimum threshold."""
    model_path = 'models/linear_regression_model.joblib'
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    model = joblib.load(model_path)

    # Reload data for testing the R^2 score
    _, X_test, _, y_test = load_data()

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Define a reasonable minimum R^2 threshold for the California Housing dataset
    MIN_R2_THRESHOLD = 0.5
    print(f"\nObserved R^2 Score: {r2:.4f}")
    assert r2 > MIN_R2_THRESHOLD, f"R^2 score {r2:.4f} did not exceed threshold {MIN_R2_THRESHOLD}"