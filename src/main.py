import os
import pandas as pd
from src.utils import clear_directory, mount_drive
from src.yolo_detection import get_dataset_insights
from src.extract_timestamps import extract_timestamps
from src.process_images import process_image, check_number_plate

def main():
    # Input paths
    dataset_path = '/content/drive/MyDrive/Intel/processed_dataset'
    images_folder = '/content/drive/MyDrive/Intel/New_Data/Photos'
    np_dataset_folder = '/content/drive/MyDrive/Intel/NP_dataset'

    # Output paths
    save_path = '/content/drive/MyDrive/Intel/detected_images'
    insights_save_path = '/content/drive/MyDrive/Intel/insights'
    excel_path = '/content/drive/MyDrive/Intel/insights/Timestamp_Validation_Results.xlsx'
    merged_excel_path = '/content/drive/MyDrive/Intel/insights/Merged_Results.xlsx'

    # Mount Google Drive
    mount_drive()

    # Clear old data
    clear_directory(save_path)
    clear_directory(insights_save_path)

    # Load a pre-trained YOLOv8 model (smallest version)
    model = YOLO('yolov8n.pt')

    # Get insights and totals
    detection_results_path, total_images, total_persons, total_bikes, total_cars, total_other_vehicles = get_dataset_insights(dataset_path, save_path, insights_save_path, model)

    # Extract timestamps from images
    timestamp_df = extract_timestamps(images_folder, excel_path)

    # Read the detection results from the previously saved Excel file
    detection_df = pd.read_excel(detection_results_path)

    # Merge the two DataFrames based on the image name
    merged_df = pd.merge(detection_df, timestamp_df, how='left', left_on='image_name', right_on='Image Name')

    # Select only the necessary columns for the final merged file
    final_merged_df = merged_df[['image_name', 'persons', 'bikes', 'cars', 'other_vehicles', 'Timestamp', 'Validation Status']]

    # Save the merged DataFrame to a new Excel file
    final_merged_df.to_excel(merged_excel_path, index=False)

    # Print insights
    print("\nTotal Images:", total_images)
    print("Total Persons:", total_persons)
    print("Total Bikes:", total_bikes)
    print("Total Cars:", total_cars)
    print("Total Other Vehicles:", total_other_vehicles)
    print("\nValidation results saved to Excel:", excel_path)
    print("Merged results saved to Excel:", merged_excel_path)

    # Calculate peak and low times
    def calculate_peak_low_times(df):
        # Calculate the total number of entities for each image
        df['total_entities'] = df['persons'] + df['bikes'] + df['cars'] + df['other_vehicles']

        # Identify the peak times (images with the most entities)
        peak_times_df = df.sort_values(by='total_entities', ascending=False).head(10)  # Top 10 peak times

        # Identify the low times (images with the fewest entities)
        low_times_df = df.sort_values(by='total_entities', ascending=True).head(10)  # Top 10 low times

        return peak_times_df, low_times_df

    # Read the final merged DataFrame from the Excel file
    merged_df = pd.read_excel(merged_excel_path)

    # Calculate peak and low times
    peak_times_df, low_times_df = calculate_peak_low_times(merged_df)

    # Prepare insights data
    insights_data = {
        'Type': ['Peak Time'] * len(peak_times_df) + ['Low Time'] * len(low_times_df),
        'Image Name': list(peak_times_df['image_name']) + list(low_times_df['image_name']),
        'Timestamp': list(peak_times_df['Timestamp']) + list(low_times_df['Timestamp']),
        'Persons': list(peak_times_df['persons']) + list(low_times_df['persons']),
        'Bikes': list(peak_times_df['bikes']) + list(low_times_df['bikes']),
        'Cars': list(peak_times_df['cars']) + list(low_times_df['cars']),
        'Other Vehicles': list(peak_times_df['other_vehicles']) + list(low_times_df['other_vehicles']),
        'Total Entities': list(peak_times_df['total_entities']) + list(low_times_df['total_entities'])
    }

    # Convert insights data to DataFrame
    insights_df = pd.DataFrame(insights_data)

    # Save the insights DataFrame to a new Excel file
    insights_file_path = os.path.join(insights_save_path, 'Peak_Low_Times_Insights.xlsx')
    insights_df.to_excel(insights_file_path, index=False)

    # Print message indicating that the insights file has been saved
    print("Peak and low times insights saved to Excel:", insights_file_path)

    # Process number plates from the dataset
    valid_number_plates = []
    partial_number_plates = []
    invalid_number_plates = []

    for split in ['train', 'val']:
        images_folder = os.path.join(np_dataset_folder, 'images', split)
        labels_folder = os.path.join(np_dataset_folder, 'labels', split)

        for image_file in os.listdir(images_folder):
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, image_file)
                label_file = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')

                if os.path.exists(label_file):
                    text = process_image(image_path)
                    print(f"Detected Text for {image_file}: {text}")
                    if text != "No text detected":
                        status = check_number_plate(text)
                        if status == "valid":
                            valid_number_plates.append(text)
                        elif status == "partial":
                            partial_number_plates.append(text)
                        else:
                            invalid_number_plates.append(text)
                else:
                    print(f"No label file found for {image_file}")

    print("Valid Number Plates:", valid_number_plates)
    print("Partially Detected Number Plates:", partial_number_plates)
    print("Invalid Number Plates:", invalid_number_plates)

if __name__ == "__main__":
    main()
