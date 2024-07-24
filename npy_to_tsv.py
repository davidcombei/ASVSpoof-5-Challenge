import numpy as np

scores = np.load('/home/asvspoof/work/ASVSpoof2024/CNN/model_scores.npy')

filenames = [f"E_{i:010}.png" for i in range(len(scores))]  

assert len(filenames) == len(scores), "Mismatch between number of filenames and scores"

results = list(zip(filenames, scores))

with open('/home/asvspoof/work/ASVSpoof2024/CNN/model_scores.tsv', 'w') as f:
    for filename, score in results:
        f.write(f"{filename}\t{score}\n")
