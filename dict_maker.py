import numpy as np
import torch

train_feats = np.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/400x768_train_wavLMV3.npy')
train_labels = np.load('/home/asvspoof/DATA/asvspoof24/train_labels_2output_27k.npy')

data = {'feats': train_feats, 'label': train_labels}
#torch.save(data,'/home/asvspoof/DATA/asvspoof24/wavLM-base/train_200x768_wavLMV3.pth')
with open('/home/asvspoof/DATA/asvspoof24/wavLM-base/train_400x768_wavLMV3.pth', 'wb') as f:
        torch.save(data, f, pickle_protocol=4)




