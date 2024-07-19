import numpy as np
import os
import torch
import torch.nn as nn

from models import ResNet32Binary, block, BiLSTM, ResNet50, ResNet50BiLSTM
import torchvision.models as models

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
         img_path = self.image_files[idx]
         image = Image.open(img_path).convert('RGB')
         image = self.transform(image)
         
         return image, os.path.split(img_path)[1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    #model = ResNet32Binary(num_classes=2)
    state_dict = torch.load("/home/asvspoof/work/ASVSpoof2024/CNN/saved_models/asv_im_ResNet50_finetune_fullSet.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    test_dataset = TestDataset('/home/asvspoof/DATA/asvspoof24/Im_E')
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False)
    print("Extracting testing features")
    X_val = np.array([2,2])
    all_files = []
    with torch.no_grad():
        for imgs, files in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            X_val = np.vstack((X_val, outputs.cpu().numpy()))
            all_files.extend(files)

    X_val = X_val[1:, :]
    print(X_val.shape, len(all_files), files[:4])
    print("Evaluating Model")
    y_val_prob = torch.softmax(torch.tensor(X_val), axis=1).numpy()

    with open("model_scores_ResNet50_finetune_fullset.txt", 'w') as fout:
        fout.write("filename\tcm-score\n")
        for k in range(X_val.shape[0]):
            fout.write(f"{all_files[k][:-4]}\t{np.round(y_val_prob[k][1],4)}\n")

if __name__ == '__main__':
    main()
