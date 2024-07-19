
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, WavLMForSequenceClassification, AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset, Audio
import numpy as np
from sklearn.metrics import roc_curve
import torchaudio
from huggingface_hub import login
from tqdm import tqdm

login()


model_name = "DavidCombei/wavLM-base-UTCN_114k"
model = AutoModelForAudioClassification.from_pretrained(model_name, force_download=True)
processor = AutoFeatureExtractor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset_asvspoof = load_dataset('DavidCombei/Wav2Vec2_ASVSpoof5_27k', split='validation')
dataset_asvspoof = dataset_asvspoof.cast_column("audio", Audio(sampling_rate=16000))


def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


y_true = []
y_scores = []

for example in tqdm(dataset_asvspoof):
    file_array = example["audio"]["array"]
    label = example["label"]

    inputs = processor(file_array, sampling_rate=16000, return_tensors="pt", padding=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    score = torch.softmax(logits, dim=1)[0, 1].item()
#    print(score)

    y_true.append(label)
    y_scores.append(score)

eer = calculate_eer(y_true, y_scores)
print(f"Equal Error Rate (EER): {eer * 100:.2f}%")


