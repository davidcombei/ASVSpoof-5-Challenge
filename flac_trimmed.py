import librosa
import os
from torch.utils.data import Dataset
import torchaudio
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

#indir = '/home/asvspoof/DATA/asvspoof24/flac_D/'
#outdir = '/home/asvspoof/DATA/asvspoof24/flac_D_trimmed/'

for fi in tqdm(sorted(os.listdir(indir))):
    wav_path = os.path.join(indir, fi)
    wav, sr = torchaudio.load(wav_path, normalize=True)
    st = wav.shape[1] - torchaudio.functional.vad(wav, sample_rate=sr, trigger_level=5).shape[1]
    en = wav.shape[1] - torchaudio.functional.vad(torch.flip(wav, [1]), sample_rate=sr, trigger_level=5).shape[1]
    wav = wav[:, st:wav.shape[1] - en]

    outpath = os.path.join(outdir,fi)
    torchaudio.save(outpath, wav, sample_rate=sr)

