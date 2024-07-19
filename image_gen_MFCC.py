import os 
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch


def get_file_names(metafile):
    with open(metafile) as fin:
         file_names = [x.split()[1] + '.flac' for x in fin.readlines()]
                
        #file_names = [x.strip() + '.flac' for x in fin.readlines()]
    return file_names


def generate_mel_images(idx, metafile,  input_dir, dataset, output_dir, colormap='viridis'):
    flac_files = get_file_names(metafile)
    flac_file = os.path.join(input_dir, dataset, flac_files[idx])
    output_image_file = os.path.join(output_dir, f'{os.path.splitext(flac_files[idx])[0]}.png')

    try:
        flac, sr = torchaudio.load(flac_file, normalize=True)
    except Exception as e:
        print(f'Failed to load {flac_file}: {e}')
        return

    transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)
    mfcc = transform(flac)

    if torch.isnan(mfcc).any(): #in case of Not a Number value inside
        mfcc[torch.isnan(mfcc)] = 0


    mfcc = mfcc.squeeze()
    plt.figure(figsize=(10,4))
    plt.imshow(mfcc, aspect='auto', origin='lower', cmap=colormap)

    plt.axis('off')

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f'Folder created: {output_dir}')
        except OSError as e:
            print(f'Failed to create directory {output_dir}: {e}')

    plt.savefig(output_image_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'File created: {os.path.join(output_dir, output_image_file)}')

input_dir = "/home/asvspoof/DATA/asvspoof24/"
dataset = "flac_D"
output_dir = "/home/asvspoof/DATA/asvspoof24/Im_MFCC_D"
#metafile = '/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.train_MEDIUM.metadata.txt"'
metafile = '/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev_MEDIUM.metadata.txt'
file_names = get_file_names(metafile)
print(f"First file name: {file_names[0]}")
print(len(file_names))

for idx in range(len(file_names)):
    generate_mel_images(idx, metafile, input_dir, dataset, output_dir)



