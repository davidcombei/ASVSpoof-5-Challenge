import os
from tqdm import tqdm
import numpy as np
import librosa
from huggingface_hub import notebook_login
from huggingface_hub import login
from datasets import Dataset, DatasetDict, Audio


#login()


def get_labels_train():
    metadata = "/home/asvspoof/DATA/114k_metadata.csv"
    labels_train = []
    file_paths = []
    count = 0
    with open(metadata, 'r') as fi:
        data = [x for x in fi.readlines()]
        for line in data:
            parts = line.strip().split()
            file_paths.append(parts[0])
            labels_train.append([1] if parts[-1] == 'bonafide' else [0])
            count+=1
    return file_paths, labels_train

def create_tuples_train(file_paths, labels_train):
    emotion_tuples = []

    assert len(file_paths) == len(labels_train), "Mismatch between file paths and labels in train"

    for x in range(len(file_paths)):
        emotion_tuples.append((file_paths[x], labels_train[x]))
        
    return emotion_tuples


file_paths_train, data_train = get_labels_train()
emotion_tuples_train = create_tuples_train(file_paths_train, data_train)

# print(emotion_tuples_train)
# print(len(emotion_tuples_train))


#### validation

def get_labels_dev():
        metadata = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev_MEDIUM.metadata.txt"
        labels_dev = []
        file_paths = []
        count = 0
        with open(metadata, 'r') as fi:
            data = [x for x in fi.readlines()]
            for line in data:
                parts = line.strip().split()
                path = '/home/asvspoof/DATA/asvspoof24/flac_D/'
                file_paths.append(f'{path}{parts[1]}.flac')  # File path
                labels_dev.append([1] if parts[-1] == 'bonafide' else [0])  # Label
                count += 1
                print(f"Number of labels and paths  extracted from the metadata: {len(labels_dev)}, {len(file_paths)}")
                # print('number of labels extracted from the metadata: ', count)
        return file_paths, labels_dev

def create_tuples_validation(file_paths, labels_dev):
    emotion_tuples = []
    
    assert len(file_paths) == len(labels_dev), "Mismatch between file paths and labels"

    for i in range(len(file_paths)):
        emotion_tuples.append((file_paths[i], labels_dev[i]))

    return emotion_tuples

file_paths_validation, data_validation = get_labels_dev()
emotion_tuples_validation = create_tuples_validation(file_paths_validation, data_validation)

                                                                                                                                    
                                                                                                                                    
                                                                                                                                    
                                                                                                                                    

# print(emotion_tuples_validation)
# print(len(emotion_tuples_validation))




CROP = 5 # seconds  -- number of seconds to retain from the audio
SAMPLING_RATE = 16_000

features_train = [feature for feature, _ in emotion_tuples_train]
labels_train = [emotion for _, emotion in emotion_tuples_train]

features_validation = [feature for feature, _ in emotion_tuples_validation]
labels_validation = [emotion for _, emotion in emotion_tuples_validation]



print("Number of training labels:", len(labels_train))
print("Number of training features:", len(features_train))
print("Number of validation labels:", len(labels_validation))
print("Number of validation features:", len(features_validation))

# Check if lengths match
if len(labels_train) != len(features_train):
        raise ValueError("Mismatch in number of training labels and features.")
if len(labels_validation) != len(features_validation):
        raise ValueError("Mismatch in number of validation labels and features.")

##################
#### TRAIN
print("TRAIN data")
NUM_TRAIN_SAMPLES = len(labels_train)

train_label2id_array = labels_train


train_file_array = [None]*NUM_TRAIN_SAMPLES
index = 0
for file_name in tqdm(features_train):
#  print('aici e eroarea')
  file_location = f'{file_name}'
#  print(file_location)
  train_file_array[index] = file_location
  
  index+=1


##################
#### VALIDATION
print("VALIDATION data")
NUM_VAL_SAMPLES = len(labels_validation)
validation_label2id_array = labels_validation
index = 0
#validation_audio_array =  np.zeros((NUM_VAL_SAMPLES, CROP*SAMPLING_RATE), dtype=np.float32)
validation_file_array = [None]*NUM_VAL_SAMPLES
for file_name in tqdm(features_validation):
  file_location = f'{file_name}'
  validation_file_array[index] = file_location
  #audio_data, _ = librosa.load(file_location, sr=None, duration=CROP)
  #validation_audio_array[index, :len(audio_data)] = audio_data
  index+=1



train_dataset = Dataset.from_dict({
    'audio': train_file_array,
    'label': train_label2id_array
}).cast_column("audio", Audio())

validation_dataset = Dataset.from_dict({
    'audio': validation_file_array,
    'label': validation_label2id_array
}).cast_column("audio", Audio())

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,

})



dataset_dict.push_to_hub('DavidCombei/WavLM-base-dataset_114k')



