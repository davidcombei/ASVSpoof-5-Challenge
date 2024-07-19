import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ASVSpoof5_im(Dataset):
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
        self.transform = transforms.ToTensor()

    def process_asv_metadata(self):

        with open(self.metafile) as fin:
            asv_labels = [[x.strip().split()[1], np.array([0.,1.]) if x.strip().split()[-1] == 'bonafide' else np.array([1.,0.])] for x in fin.readlines()]
            return asv_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = os.path.join(self.datadir, self.training_data[idx][0] + '.png')
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        #image = (image - image.min()) / (image.max() - image.min())
        
        Ys = self.training_data[idx][1]
        Ys = torch.from_numpy(Ys).float()
        return image, Ys, self.training_data[idx][0]


