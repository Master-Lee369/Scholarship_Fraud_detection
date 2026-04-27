import pytesseract
from PIL import Image
import cv2
import numpy as np
import re


def preprocess_image(image_path):
    """Enhance image for better OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("The uploaded file could not be read as an image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed


def extract_text(file_path):
    """Extract text from image or PDF using Tesseract OCR."""
    try:
        if file_path.lower().endswith('.pdf'):
            # Convert PDF pages to images first (requires pdf2image)
            try:
                from pdf2image import convert_from_path
            except ImportError:
                return "OCR unavailable: PDF support requires pdf2image and Poppler."
            pages = convert_from_path(file_path)
            full_text = ''
            for page in pages:
                full_text += pytesseract.image_to_string(page)
            return full_text
        else:
            processed = preprocess_image(file_path)
            return pytesseract.image_to_string(Image.fromarray(processed))
    except Exception as e:
        return f"OCR Error: {str(e)}"


def detect_tampering(file_path):
    """
    Detect possible image tampering using Error Level Analysis (ELA).
    Returns a tamper score between 0.0 and 1.0.
    """
    try:
        img = Image.open(file_path).convert('RGB')
        # Save at low quality and compare
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        compressed = Image.open(buffer)

        original = np.array(img, dtype=np.float32)
        compressed_arr = np.array(compressed, dtype=np.float32)

        ela = np.abs(original - compressed_arr)
        score = float(ela.mean()) / 255.0
        return round(score, 4)
    except Exception:
        return 0.0


def cross_verify_documents(application):
    """
    Compare extracted text across all documents of an application.
    Returns list of inconsistency descriptions.
    """
    docs = application.documents.all()
    extracted = {doc.doc_type: doc.extracted_text for doc in docs}
    issues = []

    # Check if name appears consistently
    name = application.full_name.lower()
    for doc_type, text in extracted.items():
        if not text or text.lower().startswith('ocr '):
            continue
        if text and name not in text.lower():
            issues.append(f"Applicant name not found in {doc_type} document.")

    # Check DOB consistency
    dob_str = application.dob.strftime('%d/%m/%Y')
    dob_patterns = [dob_str, application.dob.strftime('%Y-%m-%d')]
    for doc_type, text in extracted.items():
        if not text or text.lower().startswith('ocr '):
            continue
        if text and doc_type == 'identity':
            if not any(p in text for p in dob_patterns):
                issues.append(f"Date of birth mismatch in {doc_type} document.")

    return issues
