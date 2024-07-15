from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pandas as pd

def detect_and_save(image_path, save_dir, model):
    img = Image.open(image_path)
    results = model(img)

    # Convert results to the format we need for saving
    annotated_img = results[0].plot()

    # Save results using OpenCV
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    return results

def get_dataset_insights(dataset_path, save_dir, insights_save_dir, model):
    insights = {
        'image_name': [],
        'persons': [],
        'bikes': [],
        'cars': [],
        'other_vehicles': [],
    }

    total_images = 0
    total_persons = 0
    total_bikes = 0
    total_cars = 0
    total_other_vehicles = 0

    for phase in ['train', 'val']:
        image_dir = os.path.join(dataset_path, 'images', phase)

        for image_file in os.listdir(image_dir):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(image_dir, image_file)
                results = detect_and_save(image_path, save_dir, model)

                # Collect insights for the current image
                image_name = os.path.basename(image_file)
                persons_count = 0
                bikes_count = 0
                cars_count = 0
                other_vehicles_count = 0

                for result in results[0].boxes:
                    cls = int(result.cls[0])
                    if cls == 0:
                        persons_count += 1
                    elif cls == 1:
                        bikes_count += 1
                    elif cls == 2:
                        cars_count += 1
                    elif cls in [5, 7]:
                        other_vehicles_count += 1

                # Append the collected data to insights dictionary
                insights['image_name'].append(image_name)
                insights['persons'].append(persons_count)
                insights['bikes'].append(bikes_count)
                insights['cars'].append(cars_count)
                insights['other_vehicles'].append(other_vehicles_count)

                # Update totals
                total_images += 1
                total_persons += persons_count
                total_bikes += bikes_count
                total_cars += cars_count
                total_other_vehicles += other_vehicles_count

    # Convert insights to DataFrame
    insights_df = pd.DataFrame(insights)

    # Save DataFrame to Excel
    detection_results_path = os.path.join(insights_save_dir, 'detection_results.xlsx')
    insights_df.to_excel(detection_results_path, index=False)

    return detection_results_path, total_images, total_persons, total_bikes, total_cars, total_other_vehicles
