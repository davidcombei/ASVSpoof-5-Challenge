import os
from torch.utils.data import Dataset
import torchaudio
import torch
import numpy as np
import torch.nn.functional as F


#############
## ASV dataset
class ASVSpoof5(Dataset):
  def __init__(self, metafile, datadir, max_samples):
    self.metafile = metafile
    self.datadir = datadir
    self.max_samples = max_samples
    self.training_data = self.process_asv_metadata()
    n_samples = len(self.training_data)
    
    if max_samples > 0 and max_samples < n_samples:
      self.training_data = self.training_data[:max_samples]
      print(f"Using {max_samples} out of {n_samples}")

    self.num_samples = len(self.training_data)
    print("ASV Number of samples:", self.num_samples)

  def process_asv_metadata(self):
    with open(self.metafile) as fin:
        asv_labels = [[x.strip().split()[1], np.array([1., 0.]) if x.strip().split()[-1]=='bonafide' else np.array([0., 1.])] for x in fin.readlines()]
    return asv_labels

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    ## data processing
    wav, sr = torchaudio.load(os.path.join(self.datadir, self.training_data[idx][0]+'.flac'), normalize=True)
    transform = torchaudio.transforms.MelSpectrogram(sr, normalized=True) 
    mel_specgram = torch.log(transform(wav)+ 1e-7) # Adding a small value to avoid log(0)
    # Normalize the log mel spectrogram (optional)
    mel_specgram = (mel_specgram - mel_specgram.min()) / (mel_specgram.max() - mel_specgram.min())
    feat = mel_specgram.squeeze()

    Ys = self.training_data[idx][1]
    Ys = torch.from_numpy(Ys).float()
    return feat, Ys, self.training_data[idx][0]


###############
## FeatCollate2D
class FeatCollate2D:
    """Zero-pads model inputs and targets based on number of frames per step"""
    def __call__(self, batch):
        input_lengths = torch.LongTensor([x[0].shape[1] for x in batch])
        padded_images = []
        MAX_WIDTH = 500
        for feat, _, _ in batch:
            feat_tensor = torch.Tensor(feat)
            height, width = feat_tensor.shape
            if width < MAX_WIDTH:
                # Calculate padding
                pad_width = MAX_WIDTH - width
                # Apply padding (pad each side equally)
                padding = (pad_width//2, pad_width - pad_width//2)
                padded_image = F.pad(feat_tensor, padding, mode='constant', value=0)
            else:
                padded_image = feat_tensor[:,:MAX_WIDTH]
            padded_images.append(padded_image)

        # Stack data into a batch
        padded_images = torch.stack(padded_images)
        Ys = torch.Tensor(np.array([x[1] for x in batch]))
        files = [x[2] for x in batch]
        return (padded_images, Ys, input_lengths, files)



###############
## FeatCollate3D
class FeatCollate3D:
    def __call__(self, batch):
        input_lengths = torch.LongTensor([x[0].shape[2] for x in batch])
        padded_images = []
        MAX_WIDTH = 300
        for image,_,_ in batch:
            image_tensor = torch.Tensor(image)
            _, height, width = image_tensor.shape
            if width < MAX_WIDTH:
                # Calculate padding
                pad_width = MAX_WIDTH - width
                # Apply padding (pad each side equally)
                padding = (pad_width//2, pad_width - pad_width//2)
                padded_image = F.pad(image_tensor, padding, mode='constant', value=0)
            else:
                padded_image = image_tensor[:,:,:MAX_WIDTH]
            padded_images.append(padded_image)

        # Stack data into a batch
        padded_images = torch.stack(padded_images)
        Ys = torch.Tensor(np.array([x[1] for x in batch]))
        files = [x[2] for x in batch]
        return (padded_images, Ys, input_lengths, files)
