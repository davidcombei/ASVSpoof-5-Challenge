import numpy as np

dev_feats = np.load('/home/asvspoof/DATA/asvspoof24/feats_dev.npy')
repaired_dev = dev_feats[:140950]
np.save('/home/asvspoof/DATA/asvspoof24/feats_dev_repaired.npy', repaired_dev)
