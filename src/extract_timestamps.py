import pandas as pd
from src.process_images import process_image

def extract_timestamps(images_folder, excel_path):
    # List to store all timestamp information
    all_timestamps_info = []

    # Iterate through all images in the folder
    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(images_folder, image_file)
            print(f"Processing image: {image_path}\n")
            timestamp, status, image_name = process_image(image_path)
            if timestamp:
                print(f"Formatted Timestamp: {timestamp}, Status: {status}")
                all_timestamps_info.append((image_name, timestamp, status))
            else:
                print(f"Timestamp not found for image: {image_name}, Status: {status}")
                all_timestamps_info.append((image_name, "Not found", status))

    # Create a pandas DataFrame for timestamps
    timestamp_df = pd.DataFrame(all_timestamps_info, columns=['Image Name', 'Timestamp', 'Validation Status'])
    timestamp_df.to_excel(excel_path, index=False)

    return timestamp_df
