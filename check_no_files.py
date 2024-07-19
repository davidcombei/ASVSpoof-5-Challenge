import os


input_dir = "/home/asvspoof/DATA/asvspoof24/Im_E"
entries = os.listdir(input_dir)
file_count = len([entry for entry in entries if os.path.isfile(os.path.join(input_dir, entry)) and entry.endswith('.png')])

print(f"The number of images in {input_dir} is: {file_count}")

input_file = "/home/asvspoof/DATA/asvspoof24/EVAL_DATA/ASVspoof5.track_1.progress.trial.txt"
line_count = 0
with open(input_file) as fin:
    line_count = len([line for line in fin.readlines()])

print(f"The number of lines in {input_file} is: {line_count}")
