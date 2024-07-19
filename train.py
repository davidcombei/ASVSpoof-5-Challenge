import sys, os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Precision, Recall, ConfusionMatrix, Accuracy

from copy import deepcopy
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm


from dataset import ASVSpoof5, FeatCollate2D
from models import Conv1DModel, Transformer, GRUModel,CNNSelfAttn

import warnings
warnings.filterwarnings("ignore")


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

def compute_dcf(fpr, fnr, p_target, c_miss, c_fa):
    """Compute Detection Cost Function (DCF)"""
    return c_miss * p_target * fnr + c_fa * (1-p_target) * fpr

def compute_min_dcf(fpr, fnr, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    """Compute minimum Detection Cost Function (minDCF)"""
    dcf = compute_dcf(fpr, fnr, p_target, c_miss, c_fa)
    min_dcf = np.min(dcf)
    min_dcf_threshold = thresholds[np.argmin(dcf)]
    return min_dcf, min_dcf_threshold

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
        num_iters = 0

        for batch in tqdm(loader):
           # X, y, _, _ = batch
            X,y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)

    min_dcf, min_dcf_thresh = compute_min_dcf(fpr, fnr, thresholds)
    min_dcf_thresh = torch.Tensor([min_dcf_thresh])

    conf_matrix.threshold = eer_thresh
    print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())

    precision.threshold = eer_thresh
    recall.threshold = eer_thresh
    test_prec = precision(Y_hat_score, Y)
    test_rec = recall(Y_hat_score, Y)

    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))
    print(subset + " minDCF: ", np.round(min_dcf, 2), "| minDCF Threshold: ", np.round(min_dcf_thresh.item(), 2))
    print(f"{subset} Precision: {np.round(test_prec.item() * 100, 2)} | Recall: {np.round(test_rec.item() * 100, 2)} | ")

    ece = compute_ece(Y.cpu().detach().numpy(), Y_hat_score.cpu().detach().numpy())
    print(subset + " ECE: ", np.round(ece * 100, 2))
    return eer, eer_thresh, min_dcf, min_dcf_thresh


############
## TRAIN
def train(model, train_loader, val_loader, optimizer, lossfn, num_epochs=25, device='cuda', out_model=''):
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary", average='weighted').to(device)
    accuracy = Accuracy(task="multilabel", num_labels=50).to(device)

    MAX_EER = 1000
    print("\n*** Starting the training process...")
    for epoch in range(num_epochs):
        model.train()
        tloss = 0
        num_steps = 1
        loop = tqdm(train_loader)
        for batch in loop:
            # X, y,_,_ = batch
            X,y = batch
            X = X.to(device)
            y = y.to(device)
#            print(X.shape)
            optimizer.zero_grad()
            y_hat = model(X)

            loss = lossfn(y_hat, y)
            tloss += loss.item() / num_steps
            num_steps += 1

            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=tloss,
                             precision=precision(y_hat, y).item())  # , accuracy=accuracy(y_hat_sig, y).item())

        # VALIDATION
        eer, eer_thresh, min_dcf, min_dcf_thresh = eval(val_loader, model, device, lossfn, 'VAL')
        if eer <= MAX_EER:# and thresh.item() > 0 and thresh.item() < 1.0:
            best_model = deepcopy(model)
            MAX_EER = eer
            torch.save(best_model.state_dict(), out_model)
            print(f"Saved model to {out_model} at {np.round(eer*100,2)}")
    print('BEST EER:', MAX_EER)    
    return best_model


def main():
    ### ASVSpoof
    trainfile_asv = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.train.metadata.txt"
    valfile_asv = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev.metadata.txt"
    traindir_asv_wav = "/home/asvspoof/DATA/asvspoof24/flac_T/"
    valdir_asv_wav = "/home/asvspoof/DATA/asvspoof24/flac_D/"

    train_file = torch.load("/home/asvspoof/DATA/asvspoof24/wavLM-base/train_200x768_wavLMV3.pth")
    train_feats = train_file['feats']
    train_labels = train_file['label']

    dev_file = torch.load("/home/asvspoof/DATA/asvspoof24/wavLM-base/dev_200x768_wavLMV3.pth")
    dev_feats = dev_file['feats']
    dev_labels = dev_file['label']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING on:", device)
    
    OUT_MODEL = 'saved_models/CNNSelfAttn.pt'
    seed = 46
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ## Setup data and model
    print("*** Loading training data...")
    #traindataset = ASVSpoof5(trainfile_asv, traindir_asv_wav, max_samples=-1)

    train_DS = TensorDataset(torch.Tensor(train_feats),torch.Tensor(train_labels))                      
    trainloader = DataLoader(train_DS, batch_size =256,shuffle=True)

    dev_DS = TensorDataset(torch.Tensor(dev_feats), torch.Tensor(dev_labels))
    valloader = DataLoader(dev_DS, batch_size=256, shuffle=False)
    print("\n*** Loading validation data...")
    #valdataset = ASVSpoof5(valfile_asv, valdir_asv_wav, max_samples=-1)

   # model = Conv1DModel(input_channels=128, kernel_size=10)
   # model =Transformer(src_vocab_size=2, d_model=500, num_heads=5,num_layers=6, d_ff=2048, max_seq_length=128, dropout=0.1)
    #model = GRUModel(input_dim=500, hidden_dim=128, output_dim=2, dropout=0.2, layers=2,bidirectional_flag=True)
    model = CNNSelfAttn(embedding_dim=768,filter_sizes= [2,3],output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.93, 0.98), lr=3e-4)
    loss_ce = nn.BCEWithLogitsLoss()

    collate_fn = FeatCollate2D()
    BATCH_SIZE = 256
    #trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    #valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=False , collate_fn = collate_fn, num_workers=0)
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models/')

    ## Run the train loop
    best_model = train(model=model.to(device),
                       train_loader=trainloader,
                       val_loader=valloader,
                       optimizer=optimizer,
                       lossfn=loss_ce,
                       device=device,
                       out_model = OUT_MODEL)
    
    
    ## Save final model
    torch.save(best_model.state_dict(), OUT_MODEL[:-3]+'_final.pt')
    print("Saved final model to ... ", OUT_MODEL[:-3]+'_final.pt')

    

if __name__ == '__main__':
    main()
