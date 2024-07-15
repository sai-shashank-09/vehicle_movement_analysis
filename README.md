# Vehicle Analysis Project

## Description
This project aims to analyze vehicle movement using edge AI. It involves detecting objects in images, extracting timestamps, and generating insights.

## Setup
1. Install necessary libraries:
    ```sh
    pip install ultralytics pytesseract Pillow opencv-python openpyxl pandas
    ```
2. Install Tesseract OCR:
    ```sh
    sudo apt-get install tesseract-ocr
    ```

## Directory Structure

vehicle_analysis/
├── data/
│ ├── input/
│ │ ├──NP_dataset
│ │ └──Photos
│ └── output/
│   ├──detected_images
│   ├──insights
│   ├──New_data
│   ├──number_plates
│   └──processed_dataset
├── src/
│ ├── process_images.py
│ ├── yolo_detection.py
│ ├── extract_timestamps.py
│ ├── utils.py
│ └── main.py
├── requirements.txt
└── README.md

## Running the Project
1. Run the `main.py` script to process images and generate insights.
    ```sh
    python src/main.py
    ```

## Authors
- Intel Intern
