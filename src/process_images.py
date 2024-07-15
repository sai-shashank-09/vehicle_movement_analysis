import cv2
import pytesseract
import re
import os
import pandas as pd
import imutils
import numpy as np
import easyocr

def process_image(image_path):
    img = cv2.imread(image_path)

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to remove noise
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Perform edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours in the edged image
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop over contours to find the best approximate contour
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # Create a mask and draw contours
    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Crop the image
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Perform OCR to read text from the cropped image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        # Extract and return the text
        if result:
            return result[0][-2]

    return "No text detected"

def check_number_plate(number_plate):
    # Define patterns for a valid number plate and a partial number plate
    full_pattern = r'^[A-Z]{2}\d{2,4}[A-Z0-9\- ]{4,6}$'  # Full valid pattern
    partial_pattern = r'^[A-Z\- 0-9\- ]{1,12}$'  # Allow partial matches

    if re.match(full_pattern, number_plate):
        return "valid"
    elif re.match(partial_pattern, number_plate):
        return "partial"
    return "invalid"
