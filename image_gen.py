import os 
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch


def get_file_names(metafile):
    with open(metafile) as fin:
        file_names = [x.strip().split()[1] + '.flac' for x in fin.readlines()]
    return file_names


def generate_mel_images(idx, metafile,  input_dir, dataset, output_dir, colormap='viridis'):
    flac_files = get_file_names(metafile)
    flac_file = os.path.join(input_dir, dataset, flac_files[idx])
#    print(f'Flac file: {flac_file}')
    output_image_file = os.path.join(output_dir, f'{os.path.splitext(flac_files[idx])[0]}.png')

    try:
        flac, sr = torchaudio.load(flac_file, normalize=True)
    except Exception as e:
        print(f'Failed to load {flac_file}: {e}')
        return

    transform = torchaudio.transforms.MelSpectrogram(sr, n_mels=64, normalized=True)
    mel_specgram = torch.log(transform(flac) + 1e-7)
    mel_specgram = mel_specgram.squeeze().numpy()

    mel_specgram = (mel_specgram - mel_specgram.min()) / (mel_specgram.max() - mel_specgram.min()) #normalization
    
    plt.figure(figsize=(10,4))
    plt.imshow(mel_specgram, aspect='auto', origin='lower', cmap=colormap)
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
output_dir = "/home/asvspoof/DATA/asvspoof24/mel_im_D_ALL"
metafile = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev.metadata.txt"

file_names = get_file_names(metafile)
print(f"First file name: {file_names[0]}")
print(len(file_names))

for idx in range(len(file_names)):
    generate_mel_images(idx, metafile, input_dir, dataset, output_dir)
#entries = os.listdir('/home/asvspoof/DATA/asvspoof24/flac_T')
#png_file_count = len([entry for entry in entries if os.path.isfile(os.path.join('/home/asvspoof/DATA/asvspoof24/flac_T', entry)) and entry.endswith('.flac')])

#print(png_file_count)


