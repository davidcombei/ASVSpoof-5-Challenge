import torch
import torchaudio.functional as F
import torchaudio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] == 'bonafide':
                relevant_files.append(parts[1]+'.flac')
    return relevant_files

input_dir = '/home/asvspoof/DATA/asvspoof24/flac_T/'
output_dir = '/home/asvspoof/DATA/asvspoof24/flac_T/'
relevant_files = read_metadata('/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.train_MEDIUM.metadata.txt')

for fi in os.listdir(input_dir):
    if fi in tqdm(relevant_files):
        wav, sr = torchaudio.load(os.path.join(input_dir,fi))
        snr = torch.tensor([25])
        noise_level = 80
        noise = torch.randn_like(wav) * noise_level
        noisy_wav = F.add_noise(wav, noise, snr)
        base_name, extension = os.path.splitext(fi)
        new_file_name = f"{base_name}_noisy.flac"
        torchaudio.save(os.path.join(output_dir,new_file_name), noisy_wav, sr)
        print(f'wrote file {new_file_name} in {output_dir}')











