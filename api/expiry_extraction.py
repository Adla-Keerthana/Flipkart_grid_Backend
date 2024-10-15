import easyocr
import cv2
import re
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Pydantic model for the response
class ExpiryResponse(BaseModel):
    expiry_date: str

# Function to extract dates from text using various formats
def extract_dates_from_text(text: str):
    date_patterns = [
        r'\b(?:\d{1,2}[-/\s]?\d{1,2}[-/\s]?\d{2,4})\b',  # dd-mm-yyyy or dd/mm/yyyy
        r'\b(?:\d{4}[-/\s]?\d{1,2}[-/\s]?\d{1,2})\b',      # yyyy-mm-dd or yyyy/mm/dd
        r'\b(?:\d{1,2}\s?\w{3,9}\s?\d{2,4})\b',            # dd Month yyyy
        r'\b(?:\w{3,9}\s?\d{1,2}[,\s]?\d{2,4})\b',         # Month dd, yyyy
    ]
    
    matches = []
    for pattern in date_patterns:
        matches.extend(re.findall(pattern, text))
    
    return matches

# Function to parse and compare dates
def get_expiry_date(dates):
    parsed_dates = []

    for date in dates:
        date = date.replace(" ", "")  # Remove all spaces for consistent parsing
        try:
            if '/' in date or '-' in date:
                if len(date.split('/')) == 3:
                    parsed_date = datetime.strptime(date, "%Y/%m/%d")
                else:
                    parsed_date = datetime.strptime(date, "%d-%m-%Y") if len(date.split('-')[-1]) == 4 else datetime.strptime(date, "%d-%m-%y")
            elif ' ' in date:
                parsed_date = datetime.strptime(date, "%d %b %Y") if len(date.split()[-1]) == 4 else datetime.strptime(date, "%d %B %Y")
        except ValueError:
            continue

    return max(parsed_dates) if parsed_dates else None

# Service function to process the image and extract the expiry date
async def extract_expiry_date(image_file: UploadFile) -> str:
    try:
        # Read the image using OpenCV
        image = cv2.imdecode(np.frombuffer(await image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Use EasyOCR to do OCR on the image
        results = reader.readtext(thresh_image)

        # Combine results into a single text
        extracted_text = ' '.join([result[1] for result in results])

        # Extract all dates from the text
        dates = extract_dates_from_text(extracted_text)
        
        # Determine the expiry date from the extracted dates
        expiry_date = get_expiry_date(dates)

        if expiry_date:
            return expiry_date.strftime("%d %b %Y")  # Return in desired format
        return "No expiry date found."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expiry-extraction", response_model=ExpiryResponse)
async def expiry_extraction(image_file: UploadFile = File(...)):
    try:
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        expiry_date = await extract_expiry_date(image_file)
        return ExpiryResponse(expiry_date=expiry_date)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
