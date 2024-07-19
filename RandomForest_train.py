import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, classification_report

def compute_ece(y_true, y_pred, num_bins=15):
    """Expected calibration error for binary classifier."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    counts, _ = np.histogram(y_pred, bins)
    nonzero = counts != 0
    counts = counts[nonzero]
    p_true, p_pred = calibration_curve(y_true, y_pred, n_bins=num_bins)
    return np.sum(np.abs(p_true - p_pred) * counts) / counts.sum()

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
    BATCH_SIZE = 256
    train_file = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLMV3_train.pth')
    val_file = torch.load('/home/asvspoof/DATA/asvspoof24/wavLM-base/mean_wavLMV3_dev.pth')

    train_feats = train_file['feats']
    train_labels = train_file['label']
    val_feats = val_file['feats']
    val_labels = val_file['label']

    train_DS = TensorDataset(torch.Tensor(train_feats), torch.Tensor(train_labels))
    val_DS = TensorDataset(torch.Tensor(val_feats), torch.Tensor(val_labels))

    print("*** Loading training data...")
    train_loader = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    print("\n*** Loading validation data...")
    val_loader = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)

    print("Extracting training features")
    X_train, y_train = [], []
    for batch in tqdm(train_loader):
        X, y = batch
        X_train.append(X.detach().cpu().numpy())
        y_train.append(y.detach().cpu().numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    print("Extracting validation features")
    X_val, y_val = [], []
    for batch in tqdm(val_loader):
        X, y = batch
        X_val.append(X.detach().cpu().numpy())
        y_val.append(y.detach().cpu().numpy())
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING on:", device)

    print("*** Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=46)
    rf.fit(X_train, y_train)

    print("*** Evaluating Random Forest model...")
    y_val_pred = rf.predict(X_val)
    y_val_prob = rf.predict_proba(X_val)[:, 1]

    model_save_path = "saved_models/asv_wavLMV3_RF.pkl"
    joblib.dump(rf, model_save_path)
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
