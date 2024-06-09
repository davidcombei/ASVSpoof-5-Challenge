import numpy as np
import torch

train_feats = np.load('/home/asvspoof/DATA/asvspoof24/feats_dev_repaired.npy')
train_labels = np.load('/home/asvspoof/DATA/asvspoof24/dev_labels.npy')

data = {'feats': train_feats, 'label': train_labels}
torch.save(data,'/home/asvspoof/DATA/asvspoof24/dev_mean_data.pth')





