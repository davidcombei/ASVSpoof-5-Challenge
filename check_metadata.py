import os
def check_files_exist(metadata_file):
    missing_files = []
    with open(metadata_file, 'r') as f:
        for line in f:
            file_path = line.strip().split()[0]
            if not os.path.isfile(file_path):
                missing_files.append(file_path)

    if missing_files:
        print(f"missing files ({len(missing_files)})")
    else:
        print("all files exist")


check_files_exist('/home/asvspoof/DATA/finetune_metadata_train.csv')
