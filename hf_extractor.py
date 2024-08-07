from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
import torch
import soundfile as sf
import os, sys
from tqdm import tqdm
import numpy as np

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
            # max_length=16_000,
            # truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state
        # return outputs


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
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
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
    "wavLM-base-finetuned": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-DeepFake_UTCN"
    ),
    "wavLM-UTCN": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-UTCN"
    ),
    "wavLM-V3" : lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "DavidCombei/wavLM-base-UTCN_114k"
    ),
}



def main(indir, outdir):
    a = FEATURE_EXTRACTORS['wav2vec2-base']()
    feats = []
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    available_files = os.listdir(outdir)
    for fi in tqdm(sorted(os.listdir(indir))):
        if fi not in available_files:
            audio, sr = sf.read(os.path.join(indir,fi))
            p = a(audio, sr)
            p = torch.mean(p, dim=1)
            feats.append(p.cpu().numpy())
            
    feats = np.vstack(feats)
    np.save(os.path.join(outdir, 'eval_feats_wav2vec2-finetuned.npy'), feats)
    
                            

if __name__ == '__main__':
    main('/home/asvspoof/DATA/asvspoof24/EVAL_DATA/flac_E_prog/','/home/asvspoof/DATA/asvspoof24/wav2vec2-base-finetuned')
