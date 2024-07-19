import logging

logging.basicConfig(level=logging.DEBUG)

import torch
import torchaudio
import os
from tqdm import tqdm


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] == 'bonafide':
                relevant_files.append(parts[1] + '.flac')
    return relevant_files


def apply_codec(waveform, sample_rate, format, encoder=None):
    try:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        effector = torchaudio.io.AudioEffector(format=format, encoder=encoder)

        return effector.apply(waveform, sample_rate)
    except Exception as e:
        logging.error(f"Failed to apply codec: {e}")
        return None


input_dir = '/home/asvspoof/DATA/asvspoof24/flac_T/'
output_dir = '/home/asvspoof/DATA/asvspoof24/flac_T/'
relevant_files = read_metadata('/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.train_MEDIUM.metadata.txt')

for fi in tqdm(os.listdir(input_dir)):
    if fi in relevant_files:
        wav, sr = torchaudio.load(os.path.join(input_dir, fi), channels_first=False)
        mulaw = apply_codec(wav, sr, "flac", encoder="pcm_mulaw")
        if mulaw is not None:
            base_name, extension = os.path.splitext(fi)
            new_file_name = f"{base_name}_codec.flac"
            torchaudio.save(os.path.join(output_dir, new_file_name), mulaw, sr)
            print(f'Wrote file {new_file_name} in {output_dir}')
        else:
            print(f"Failed to process file {fi}")


