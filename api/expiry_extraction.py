from fastapi import APIRouter, HTTPException, UploadFile, File
from models.response_models import ExpiryResponse
import re
from paddleocr import PaddleOCR
from io import BytesIO
from PIL import Image
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

router = APIRouter()

# Function to extract expiry date from text
def extract_expiry_date_from_text(text: str) -> str:
    # List of regex patterns to capture various date formats
    date_patterns = [
        # dd-mm-yyyy or dd/mm/yyyy
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        # dd Mon yyyy (e.g., 01 Jan 2024)
        r'\b(\d{1,2} \w{3} \d{2,4})\b',
        # Mon dd, yyyy (e.g., Jan 01, 2024)
        r'\b(\w{3} \d{1,2}, \d{2,4})\b',
        # yyyy-mm-dd or yyyy/mm/dd
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        # mm-dd-yyyy or mm/dd/yyyy
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        # Full date formats (e.g., January 1, 2024)
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',
        # Full month-day-year (e.g., 1st January 2024)
        r'\b(\d{1,2}(st|nd|rd|th) [A-Za-z]+ \d{4})\b',
        # ISO format (e.g., 2024-01-01)
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # Additional formats can be added here
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return "No expiry date found."

# Service function to process the image and extract the expiry date
async def extract_expiry_date(image_file: UploadFile) -> str:
    # Load the image using PIL
    try:
        image = Image.open(image_file.file).convert("RGB")
        logger.info("Image loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Perform OCR on the image
    try:
        # Convert image to a format suitable for PaddleOCR
        image_np = np.array(image)
        result = ocr.ocr(image_np, cls=True)
        logger.info("OCR processing completed successfully.")
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    # Combine all detected texts into a single string
    extracted_text = " ".join(word_info[1][0] for line in result for word_info in line)
    logger.info(f"Extracted text: {extracted_text}")

    # Extract the expiry date from the combined text
    expiry_date = extract_expiry_date_from_text(extracted_text)
    return expiry_date

@router.post("/expiry-extraction", response_model=ExpiryResponse)
async def expiry_extraction(image_file: UploadFile = File(...)):
    try:
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        expiry_date = await extract_expiry_date(image_file)
        return ExpiryResponse(expiry_date=expiry_date)
    except HTTPException as http_ex:
        raise http_ex  # Preserve HTTP exceptions
    except Exception as e:
        logger.error(f"Error in expiry extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
