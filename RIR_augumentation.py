import torch
import torchaudio.functional as F
import torchaudio
import os
from tqdm import tqdm
from torchaudio.utils import download_asset

SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)

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

for fi in tqdm(os.listdir(input_dir)):
    if fi in relevant_files:
        speech, sr = torchaudio.load(os.path.join(input_dir,fi))
        augmented = F.fftconvolve(speech, rir)
        base_name, extension = os.path.splitext(fi)
        new_file_name = f"{base_name}_RIR.flac"
        torchaudio.save(os.path.join(output_dir,new_file_name), augmented, sample_rate=sr)
        print(f'wrote file {new_file_name} in {output_dir}')





















