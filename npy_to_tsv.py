import numpy as np

# Load the npy file
scores = np.load('/home/asvspoof/work/ASVSpoof2024/CNN/model_scores.npy')

# Assuming you have a list of filenames corresponding to the scores
# You need to replace this with the actual code to get filenames
filenames = [f"E_{i:010}.png" for i in range(len(scores))]  # Example filenames

# Ensure filenames and scores have the same length
assert len(filenames) == len(scores), "Mismatch between number of filenames and scores"

# Combine filenames and scores
results = list(zip(filenames, scores))

# Save to a TSV file
with open('/home/asvspoof/work/ASVSpoof2024/CNN/model_scores.tsv', 'w') as f:
    for filename, score in results:
        f.write(f"{filename}\t{score}\n")
