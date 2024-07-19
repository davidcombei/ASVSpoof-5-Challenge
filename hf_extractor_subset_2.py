import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
    "wav2vec2-base-finetuned-MEDIUM": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "DavidCombei/wav2vec2-base-ASVSpoof5_MEDIUM_finetuned"
    ),
    "wavLM-V1": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-DeepFake_UTCN"
    ),
    "wavLM-V2" : lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-UTCN"
    ),
    "wavLM-V3": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-UTCN_114k"
    ),
}


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                relevant_files.append(parts[1])
    return sorted(relevant_files)


def main(outdir, metadata_file,indir):
    relevant_files = read_metadata(metadata_file)
    feature_extractor = FEATURE_EXTRACTORS['wavLM-V3']()
    feats = []

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f"{len(relevant_files)} files found in metadata")

    for fi in tqdm(relevant_files):
        if fi in relevant_files:
            file_name = f'{os.path.join(indir,fi)}.flac'
            audio, sr = sf.read(file_name)
            max_duration = 8
            max_samples = int(sr * max_duration)
            if len(audio) < max_samples:
                 padding = max_samples - len(audio)

                 audio = np.pad(audio, (padding // 2, padding - padding // 2), 'constant')
            else:
                 start = (len(audio) - max_samples) // 2
                 audio = audio[start:start + max_samples]
            p = feature_extractor(audio, sr)
            feats.append(p.cpu().numpy())

    if feats:
        feats = np.vstack(feats)
        np.save(os.path.join(outdir, '400x768_dev_wavLMV3.npy'), feats)




if __name__ == '__main__':

    indir = '/home/asvspoof/DATA/asvspoof24/flac_D/'
    outdir = '/home/asvspoof/DATA/asvspoof24/wavLM-base'
    metadata_file = '/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev_MEDI\
UM.metadata.txt'
    main(outdir, metadata_file,indir)

