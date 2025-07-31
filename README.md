MLOps Major Assignment – Linear Regression Pipeline

This repository implements a complete MLOps pipeline using **Linear Regression** on the **California Housing Dataset**, including training, testing, manual quantization, Dockerization, and CI/CD integration via GitHub Actions.

Assignment by: Akshay Kumar - G24AI1033 
Course: ML Ops – IIT Jodhpur  
Report added for reference


## Objective

To build a complete, production-ready MLOps pipeline with:

- Model training using `scikit-learn`’s `LinearRegression`
- Manual quantization of model parameters to `uint8`
- Unit tests using `pytest`
- Dockerfile for containerization
- CI/CD using GitHub Actions
- Organized and reproducible codebase

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pytest_cache/
├── models/
│   └── linear_regression_model.joblib
├── parameters/
│   ├── quant_params.joblib
│   └── unquant_params.joblib
├── src/
|   ├── _pycache_
│   ├── __init__.py
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── tests/
|   ├── _pycache_
│   ├── __init__.py
│   └── test_train.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Comparison Table

| Metric                | Original Model | Quantized Model |
|-----------------------|----------------|------------------|
| **R² Score**          | 0.5758         | 0.4376           |
| **Mean Squared Error**| 0.5559         | 0.7370           |
| **File Size**         | 681 Bytes      | 403 Bytes        |

Quantization was done manually using symmetric scaling to `uint8`, with the intercept preserved in `float32` for better accuracy.

After trying multiple quantization strategies (global, per-coefficient, symmetric scaling), we found that global symmetric quantization with clipped small coefficients gave the best balance between model size and accuracy. 

The final quantized model achieved an R² score of 0.4376 (original: 0.5758), with acceptable prediction quality.

---

## How to Run

### 1. Create and activate environment

```bash
python -m venv mlops-env
source mlops-env/bin/activate  
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train
```

### 3. Quantize the model

```bash
python -m src.quantize
```

### 4. Predict using saved model

```bash
python -m src.predict
```

---

## Run Tests

```bash
pytest
```

---

## Build and Run Docker Container

```bash
docker build -t mlops-lr .
docker run mlops-lr
```

---

## CI/CD (GitHub Actions)

The `.github/workflows/ci.yml` pipeline runs the following on every push to `main`:

1. `test suite` – Run all unit tests
2. `train and quantize` – Train + quantize model and save artifacts
3. `build and test container` – Build Docker image and run `predict.py` successfully

---

## Dataset

Used: `sklearn.datasets.fetch_california_housing()`

---

## Submission Details

- All work is in the **`main`** branch.
- All steps implemented via command line
- Link submitted through [Google Form](https://forms.gle/ANbjTHzanZwwZj8n6).
- No plagiarism or hardcoded values.
- Report added for reference

---