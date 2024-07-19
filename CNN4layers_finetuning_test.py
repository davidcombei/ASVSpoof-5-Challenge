import datetime
import torch
import numpy as np
import os, sys
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from copy import deepcopy
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm
from torchmetrics import Precision, Recall, ConfusionMatrix, Accuracy
import warnings
import random
warnings.filterwarnings("ignore")


########################
# Define the CNN model
class Conv1DModel(nn.Module):
    def __init__(self, input_channels, kernel_size=20):
        super(Conv1DModel, self).__init__()
        self.conv1d =   nn.Conv1d(input_channels, 1000, kernel_size=20, padding='same')
        self.conv1d_2 = nn.Conv1d(1000, 500, kernel_size=kernel_size//2, padding='same')#, dilation=2)
        self.conv1d_3 = nn.Conv1d(500, 250, kernel_size=5, padding='same')
        self.conv1d_4 = nn.Conv1d(250, 1, kernel_size=2, padding='same')



        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.relu(x)
        x = self.conv1d_4(x)
        x = x.squeeze(-1)
        return x





def compute_ece(y_true, y_pred, num_bins=15):
    """Expected calibration error for binary classifier."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    counts, _ = np.histogram(y_pred, bins)
    # remove bins with no samples, to match scikit-learn implementation
    nonzero = counts != 0
    counts = counts[nonzero]
    p_true, p_pred = calibration_curve(y_true, y_pred, n_bins=num_bins)
    return np.sum(np.abs(p_true - p_pred) * counts) / counts.sum()


def mask_from_lens(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


##################
## EVAL
def eval(loader, model, device, loss_fn, subset):
    precision = Precision(task="binary", average='weighted').to(device)
    recall = Recall(task="binary", average='weighted').to(device)
    accuracy = Accuracy(task="binary").to(device)
    conf_matrix = ConfusionMatrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)

        for batch in tqdm(loader):
            X, y = batch
            X = X.unsqueeze(1).to(device)
           # X = torch.as_tensor(X.permute(0, 2, 1), dtype=torch.float, device=device)
          #  y = torch.as_tensor(y, dtype=torch.int, device=device)
         #   y = y.unsqueeze(1).to(device)

        #    y_hat = model(X)
       #     preds = nn.Sigmoid()(y_hat.squeeze())

      #      Y = torch.cat((Y, y.view(-1)))
     #       Y_hat_score = torch.cat((Y_hat_score, preds.view(-1)))

   # fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
    #                                 Y_hat_score.cpu().detach().numpy(), pos_label=1)
            X = torch.as_tensor(X.permute(0, 2, 1), dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))
            
            y_hat = model(X)
           # print('model output:',y_hat)
            sig = nn.Sigmoid()
            
            y_hat = sig(y_hat)
           # print('model output after softmax:',y_hat)
           # Y_hat_score = torch.cat((Y_hat_score, y_hat[:,1]))
            Y_hat_score = torch.cat((Y_hat_score, y_hat))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
    Y_hat_score = Y_hat_score.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(Y, Y_hat_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    thresh = torch.Tensor(thresh)
    conf_matrix.threshold = thresh.item()  # Set threshold for confusion matrix
    conf_matrix.update(torch.Tensor(Y_hat_score).to(device), torch.Tensor(Y).to(device))
    cm = conf_matrix.compute().cpu().detach().numpy()
    print(cm)
    precision.threshold = thresh
    recall.threshold = thresh
    test_prec = precision(torch.Tensor(Y_hat_score).to(device), torch.Tensor(Y).to(device))
    test_rec = recall(torch.Tensor(Y_hat_score).to(device), torch.Tensor(Y).to(device))

    print(subset + " EER: ", np.round(eer * 100, 2), "| Threshold: ", np.round(thresh.item(), 2))
    print(f"{subset} Precision: {np.round(test_prec.item() * 100, 2)} | Recall: {np.round(test_rec.item() * 100, 2)} | ")

    ece = compute_ece(Y, Y_hat_score)
    print(subset + " ECE: ", np.round(ece * 100, 2))
    return eer, ece, thresh


################
##TRAIN
def train(model, train_loader, val_loader, optimizer, lossfn, num_epochs=50, device='cuda', out_path=None):
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary", average='weighted').to(device)
    accuracy = Accuracy(task="binary").to(device)

    MIN_EER = 1000
    for epoch in range(num_epochs):
        print("Started at: ", datetime.datetime.now())
        model.train()
        tloss = 0
        num_steps = 1
        loop = tqdm(train_loader)
        for batch in loop:
            X, y = batch
               
            X = X.unsqueeze(1).to(device)
            X = X.permute(0, 2, 1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            print(X.shape)
            y_hat = model(X)
            sig = nn.Sigmoid()
           # y_hat_test = sig(y_hat)
           # print('model output with sigmoid:', y_hat_test)
            loss = lossfn(y_hat, y)
            tloss += loss.item() / num_steps
            num_steps += 1

            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=tloss, precision=precision(y_hat, y).item())

        # VALIDATION
        eer, ece, thresh = eval(val_loader, model, device, lossfn, 'VAL')
        if eer <= MIN_EER and thresh.item() > 0 and thresh.item() < 1.0:
            best_model = deepcopy(model)
            print(f"New EER: {np.round(eer * 100, 2)} vs old EER: {np.round(MIN_EER * 100, 2)}. Saving model...")
            MIN_EER = eer
            if out_path:
                torch.save(best_model.state_dict(), out_path)
                print("Saved model to ... ", out_path)
    torch.save(model.state_dict(), out_path[:-3] + "_FINAL_MODEL.pt")
    return best_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING on:", device)

    seed = 46
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    BATCH_SIZE = 256

    train_file = torch.load('/home/asvspoof/DATA/asvspoof24/wav2vec2-base-finetuned/mean_MFCC_data_27k_train.pth')
    val_file = torch.load('/home/asvspoof/DATA/asvspoof24/wav2vec2-base-finetuned/mean_MFCC_data_27k_dev.pth')
   # train_file = torch.load('/home/asvspoof/DATA/asvspoof24/wav2vec2-base-finetuned/mean_data_train.pth')
   # val_file = torch.load('/home/asvspoof/DATA/asvspoof24/wav2vec2-base-finetuned/mean_data_dev.pth')

    train_feats = train_file['feats']
   train_labels = train_file['label']

    val_feats = val_file['feats'][:140950]

    val_labels = val_file['label']
    print(val_feats.shape)
    train_DS = TensorDataset(torch.Tensor(train_feats), torch.Tensor(train_labels))
    val_DS = TensorDataset(torch.Tensor(val_feats), torch.Tensor(val_labels))

    train_loader = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)

    input_channels = 768  # X.shape[2]
    model = Conv1DModel(input_channels, kernel_size=20).float()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.93, 0.98), lr=3e-4)
    loss_ce = nn.BCEWithLogitsLoss()

    OUT_MODEL = 'saved_models/CNN_1output_wav2vec2.0_finetuned.pt'
    print("Will be saving model to: ", OUT_MODEL)
    best_model = train(model=model.to(device),
                       train_loader=train_loader,
                       val_loader=val_loader,
                       optimizer=optimizer,
                       lossfn=loss_ce,
                       device=device,
                       out_path=OUT_MODEL)

    torch.save(best_model.state_dict(), OUT_MODEL)
    print("Saved model to ... ", OUT_MODEL)


if __name__ == '__main__':
    main()
