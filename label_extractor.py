import os
import numpy as np


f_in = "../../DATA/asvspoof24/metadata/ASVspoof5.dev.metadata.txt"

with open(f_in, 'r') as fi:
    data = [x for x in fi.readlines()]

labels = np.zeros((len(data),2))
for i, line in enumerate(data):
    labels[i,:] = np.array([0., 1.]) if line.strip().split()[-1] == 'bonafide' else np.array([1.,0.])

np.save("dev_labels.npy", labels)


    
