import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, WavLMForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_curve
import torchaudio
from huggingface_hub import login
from tqdm import tqdm
#login()


model_name = "DavidCombei/wavLM-base-UTCN_114k"
model = WavLMForSequenceClassification.from_pretrained(model_name, force_download=True)
processor = Wav2Vec2Processor.from_pretrained(model_name)


dataset_asvspoof = load_dataset('DavidCombei/WavLM-base-dataset_114k', split='validation')

# Calculate EER
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

y_true = []
y_scores = []
for example in tqdm(dataset_asvspoof):
    file_path = example["file"]
    label = example["label"]
    wav, sampling_rate = torchaudio.load(file_path)
    inputs = processor(wav, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
      logits_pred = model(**inputs).logits
    score = torch.sigmoid(logits_pred).item()
    y_scores.append(score)
    y_true.append(label)

eer = calculate_eer(y_true, y_scores)
print(f"EER: {eer * 100:.2f}%")
