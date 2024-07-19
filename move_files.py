import os
import shutil


current_directory = '/home/asvspoof/DATA/finetune_dev_flac/'
metadata = '/home/asvspoof/DATA/finetune_metadata_dev.csv'

def copy_files_back(metadata_file, current_dir):
    with open(metadata_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            original_path = parts[0]
            file_name = os.path.basename(original_path)
            current_path = os.path.join(current_dir, file_name)

            if os.path.isfile(current_path):
                shutil.copy(current_path, original_path)
                print(f"Copied {current_path} to {original_path}")
            else:
                print(f"File does not exist: {current_path}")

copy_files_back(metadata, current_directory)
