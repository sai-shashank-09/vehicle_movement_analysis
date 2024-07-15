import os
from google.colab import drive

# Function to clear a directory
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.makedirs(dir_path)

# Mount Google Drive
def mount_drive():
    drive.mount('/content/drive')
