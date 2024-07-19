from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import torch
import numpy as np
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf
import librosa
from huggingface_hub import login
from datasets import load_dataset
from datasets import load_metric
import warnings
warnings.filterwarnings("ignore")


#login()

dataset_asvspoof = load_dataset('DavidCombei/WavLM-base-dataset_114k')


model_checkpoint = 'microsoft/wavLM-base'
batch_size = 8
metric = load_metric("accuracy")

train_labels = dataset_asvspoof['train']['label']
validation_labels=dataset_asvspoof['validation']['label']
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

max_duration = 5

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
    )
    return inputs

encoded_dataset = dataset_asvspoof.map(preprocess_function, remove_columns=["audio"], batched=True)




num_labels = 2
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,

)

model_name = model_checkpoint.split("/")[-1]


args = TrainingArguments(
    f"{model_name}-UTCN_114k",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

#train
trainer.train()
#eval
trainer.evaluate()
#save
trainer.push_to_hub()


