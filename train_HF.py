import sys, os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, ConfusionMatrix, Accuracy
from torchvision.transforms import transforms

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, AutoModelForImageClassification

from im_dataset import ASVSpoof5_im  # Assuming you have defined this dataset class
from models import ResNet, BasicBlock, ResNetNew, block, BiLSTM, ResNet50, ResNet50BiLSTM  # Assuming these are custom models you might use

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
            X, y, _ = batch
            X = X.to(device)
            y = y.to(device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            outputs = model(X)
            probabilities = nn.Softmax(dim=1)(outputs)
            y_hat_score = probabilities[:, 1]
            y_hat_class = torch.argmax(probabilities, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            Y_hat_score = torch.cat((Y_hat_score, y_hat_score))

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

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=50, device='cuda', out_model=''):
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
            X, y, files = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            tloss += loss.item() / num_steps
            num_steps += 1

            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=tloss,
                             precision=precision(outputs, y).item())

        # VALIDATION
        eer, eer_thresh, min_dcf, min_dcf_thresh = eval(val_loader, model, device, loss_fn, 'VAL')
        if eer <= MAX_EER:
            best_model = deepcopy(model)
            MAX_EER = eer
            torch.save(best_model.state_dict(), out_model)
            print(f"Saved model to {out_model} at {np.round(eer*100,2)}")

    return best_model

def main():
    trainfile_asv = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.train_MEDIUM.metadata.txt"
    valfile_asv = "/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev_MEDIUM.metadata.txt"
    traindir_asv_wav = "/home/asvspoof/DATA/asvspoof24/Im_TT"
    valdir_asv_wav = "/home/asvspoof/DATA/asvspoof24/Im_D"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING on:", device)
    
    OUT_MODEL = '/home/asvspoof/work/ASVSpoof2024/CNN/saved_models/asv_im_ResNet18_finetune_HF_subset.pt'
    seed = 46
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    print("*** Loading training data...")
    traindataset = ASVSpoof5_im(trainfile_asv, traindir_asv_wav, max_samples=-1)

    print("\n*** Loading validation data...")
    valdataset = ASVSpoof5_im(valfile_asv, valdir_asv_wav, max_samples=-1)

    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18", num_labels=2, ignore_mismatched_sizes=True)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.93, 0.98), lr=3e-4)
    loss_ce = nn.CrossEntropyLoss()

    BATCH_SIZE = 64

    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models/')

    best_model = train(model=model.to(device),
            train_loader=trainloader,
            val_loader=valloader,
            optimizer=optimizer,
            loss_fn=loss_ce,
            num_epochs=50,
            device=device,
            out_model=OUT_MODEL)
    torch.save(best_model.state_dict(), OUT_MODEL[:-3]+'_final.pt')
    print("Saved final model to ...", OUT_MODEL[:-3]+'_final.pt')
    print("\n*** Evaluating the final model on validation data...")
    best_model.load_state_dict(torch.load(OUT_MODEL))
    eval(valloader, best_model.to(device), device, loss_ce, 'FINAL VAL')

if __name__ == '__main__':
    main()
