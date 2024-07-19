import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys, os

import joblib
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Precision, Recall, ConfusionMatrix, Accuracy

from copy import deepcopy
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, precision_score, \
    recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
#from sklearn.svm import SVC


#from dataset_CNN_ChromaSpecs import ASVSpoof5, FeatCollate2D

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


def main():
    ### ASVSpoof

    BATCH_SIZE = 100
#    train_file = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLMV3_train.pth')
 #   train_file_2 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM-finetuned_data_train.pth')
  #  train_file_3 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM-finetuned_train.pth')
  #  train_file_4 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM_data_train.pth')
  #  val_file = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLMV3_dev.pth')
  #  val_file_2 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM-finetuned_data_dev.pth')
  #  val_file_3 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM-finetuned_dev.pth')
  #  val_file_4 = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLM_data_dev.pth')
    

    train_t = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/train_400x768_wavLMV3.pth')
    val_t = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/dev_400x768_wavLMV3.pth')
    
    train_feats = train_t['feats']
    #train_labels = train_t['label']
    y_values = [label[1] for label in val_t['label']]
    train_labels = torch.tensor(y_values)
    print(train_labels.shape)
    val_feats = val_t['feats']
    #val_labels = val_t['label']
    y_values2 = [label[1] for label in val_t['label']]
    val_labels = torch.tensor(y_values2)
    
    
   # train_feats = train_file['feats']
   # train_labels = train_file['label']

  #  train_feats2 = train_file_2['feats']
  #  train_labels2 = train_file_2['label']

  #  train_feats3 = train_file_3['feats'][:27000]
  #  train_labels3 = train_file_3['label'][:27000]

  #  train_feats4 = train_file_4['feats'][:27000]
  #  train_labels4 = train_file_4['label'][:27000]

    #val_feats = val_file['feats']
    #val_labels = val_file['label']

  #  val_feats_2 = val_file_2['feats']
  #  val_labels_2 = val_file_2['label']

  #  val_feats_3 = val_file_3['feats']
  #  val_labels_3 = val_file_3['label']

  #  val_feats_4 = val_file_4['feats']
  #  val_labels_4 = val_file_4['label']

  #  print(train_feats.shape, train_feats2.shape, train_feats3.shape, train_feats4.shape)
    
  #  combined_features = np.hstack((train_feats, train_feats2, train_feats3, train_feats4))
  #  combined_labels = np.concatenate((train_labels, train_labels2, train_labels3, train_labels4),axis=0)
  #  print(combined_features.shape, combined_labels.shape)
  #  full_train_DS = TensorDataset(torch.Tensor(combined_features), torch.Tensor(train_labels))

  #  combined_features_val = np.hstack((val_feats, val_feats_2, val_feats_3, val_feats_4))

  #  full_val_DS = TensorDataset(torch.Tensor(combined_features_val), torch.Tensor(val_labels))
    

  #  combine_feats = np.concatenate((train_feats, val_feats), axis=0)
  #  combine_labels = np.concatenate((train_labels, val_labels), axis=0)

  #  combined_DS = TensorDataset(torch.Tensor(combine_feats), torch.Tensor(combine_labels))
    
    train_DS = TensorDataset(torch.Tensor(train_feats), torch.Tensor(train_labels))
    val_DS = TensorDataset(torch.Tensor(val_feats), torch.Tensor(val_labels))

  #  combined_loader = DataLoader(combined_DS, batch_size=BATCH_SIZE, shuffle=True)

  #  full_loader = DataLoader(full_train_DS, batch_size=BATCH_SIZE, shuffle=True)
  #  full_loader_val = DataLoader(full_val_DS, batch_size=BATCH_SIZE, shuffle=False)
    
    print("*** Loading training data...")
    train_loader = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    print("\n*** Loading validation data...")
    val_loader = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)



  #  print("Extracting combined features")
  #  X_combined, y_combined = [],[]
  #  for batch in tqdm(full_loader):
  #      X,y = batch
  #      X_combined.append(X.detach().cpu().numpy())
  #      y_combined.append(y.detach().cpu().numpy())

  #  X_combined = np.concatenate(X_combined,axis=0)
  #  print(X_combined.shape)
  #  y_combined = np.concatenate(y_combined,axis=0)
  #  print(y_combined.shape)



    print("Extracting training features")
    X_train, y_train = [], []
    for batch in tqdm(train_loader):
        X, y = batch
        #print(X)
#        print(X.shape)
        X = torch.mean(X,dim=1)
#        print(y)
        #print(X)
        X_train.append(X.detach().cpu().numpy())
        y_train.append(y.detach().cpu().numpy())
        #print(X_train)

    X_train = np.concatenate(X_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)
    print(X_train.shape)

    print("Extracting validation features")
    X_val, y_val = [], []
    #for batch in tqdm(val_loader):
    for batch in tqdm(val_loader):
        X, y = batch
        X = torch.mean(X, dim=1)
        X_val.append(X.detach().cpu().numpy())
        y_val.append(y.detach().cpu().numpy())
    X_val = np.concatenate(X_val,axis=0)
    y_val = np.concatenate(y_val,axis=0)
 #   print(X_val.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING on:", device)


    print("*** Training LogReg model...")
    logreg = LogisticRegression(random_state=46, C=1e3 ,max_iter=10000)
    logreg.fit(X_train, y_train)


    print("*** Evaluating LogReg model...")
    y_val_pred = logreg.predict(X_val)
    y_val_prob = logreg.predict_proba(X_val)[:, 1]

    model_save_path = "saved_models/asv_wavLMV3_200x768.pkl"
    joblib.dump(logreg , model_save_path)
    print(f"Model saved to {model_save_path}")

    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer_thresh = interp1d(fpr, thresholds)(eer)
    min_dcf, min_dcf_thresh = compute_min_dcf(fpr, fnr, thresholds)


    print(f"Validation EER: {eer*100:.2f}")
    print(f"Validation EER Threshold: {eer_thresh:.4f}")
    print(f"Validation minDCF: {min_dcf:.4f}")
    print(f"Validation minDCF Threshold: {min_dcf_thresh:.4f}")

    ece = compute_ece(y_val, y_val_prob)
    print(f"Validation ECE: {ece:.4f}")

    cnf_matrix = metrics.confusion_matrix(y_val, y_val_pred)
    print(cnf_matrix)




if __name__ == '__main__':
    main()
