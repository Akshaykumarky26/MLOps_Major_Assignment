import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import load_data
import os

def train_model():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred) 

    print(f"R^2 Score: {r2:.4f}") 
    print(f"Mean Squared Error (Loss): {mse:.4f}") 

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()