# Scholarship Fraud Detection

Scholarship Fraud Detection is a Django-based application for collecting scholarship applications, verifying uploaded documents, flagging suspicious submissions, and supporting admin review workflows.

## Technology Stack

### Machine Learning

- `scikit-learn` is the current fraud scoring backend used in the project.
- `TensorFlow` or `PyTorch` can be introduced later if we expand from anomaly detection into deeper document or image models.

### Data Processing

- `pandas`
- `numpy`

### Computer Vision

- `opencv-python-headless` for image preprocessing and tamper-oriented analysis.

### OCR

- `pytesseract` as the Python wrapper for Tesseract OCR.
- `pdf2image` for extracting OCR input from uploaded PDF files.

### Visualization

- `matplotlib`
- `seaborn`

## Current ML/OCR Usage

- Fraud scoring lives in [apps/fraud_detection/ml_scorer.py](apps/fraud_detection/ml_scorer.py) and now supports a supervised `scikit-learn` multiclass classifier (`genuine`, `suspicious`, `fake`) with a stronger rule-based fallback.
- OCR and document preprocessing live in [apps/verification/ocr_engine.py](apps/verification/ocr_engine.py) and currently use OpenCV plus Tesseract OCR.

## Requirement Coverage

The current system covers the requested areas as follows:

1. First-level validation
   - Form validation enforces data shape rules such as Aadhaar length, percentage range, file type, file size, and duplicate document uploads.
   - Runtime scoring also considers missing documents, risky submission timing, reused identifiers/IPs, and suspicious income-grade combinations.
2. Document verification
   - OCR extraction, tamper-oriented image analysis, and cross-document consistency checks are implemented.
3. Fraud detection
   - Duplicate checks, document tamper checks, and a trained multiclass classifier all contribute to the fraud score.
4. Classification
   - Applications are categorized into `genuine`, `suspicious`, or `fake`, and suspicious/fake predictions create an `ml_anomaly` review flag.

## External Data And Training

The repository now includes a reproducible training pipeline driven by open scholarship-domain data:

- External scholarship seed dataset: `data/external/scholarships_data.csv`
- Source URL: `https://huggingface.co/datasets/UmairT/scholarships_dataset`
- Training command:

  ```powershell
  .\myenv\Scripts\python.exe manage.py train_fraud_model --samples 8000
  ```

What the training step does:

- Downloads the scholarship CSV if it is missing.
- Uses that open dataset to seed scholarship competitiveness profiles.
- Generates an application-level training frame aligned to the current Django feature set.
- Trains and evaluates a supervised fraud classifier.
- Saves artifacts to `ml_models/fraud_classifier.pkl` and `ml_models/training_metrics.json`.

Why the training data is synthetic after download:

- Public scholarship datasets generally describe scholarship programs, not labeled fraudulent applications.
- The project therefore uses open scholarship data as domain context, then generates application-level fraud patterns that match the fields already collected by the app.

## Setup Notes

1. Install the Python dependencies:

   ```powershell
   .\myenv\Scripts\pip install -r requirements.txt
   ```

2. Install the Tesseract OCR desktop binary separately.

   On Windows, install Tesseract and ensure the `tesseract.exe` location is available in your system `PATH`.

3. If you want PDF OCR support, install Poppler as well so `pdf2image` can render PDF pages.

## Why TensorFlow/PyTorch Are Not Default Yet

The current implementation is built around tabular fraud scoring plus OCR-assisted document review, which `scikit-learn` handles well with much lower setup cost. Adding both TensorFlow and PyTorch right now would increase environment size and install complexity without changing runtime behavior yet.
