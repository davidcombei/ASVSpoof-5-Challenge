from torchvggish import vggish, vggish_input

# embedding_model = vggish()
# embedding_model.eval()
#
# example = vggish_input.wavfile_to_examples("flac_T/T_0000000100.flac")
# embeddings = embedding_model.forward(example)
# print(embeddings.shape)


#from leaf_pytorch.frontend import Leaf
import numpy as np
import torch
import torchaudio
import os
import pickle
from tqdm import tqdm
#rom models.classifier import Classifier


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                relevant_files.append(parts[1] + '.flac')
    return relevant_files


def main(indir, outdir, metadata_file):
    relevant_files = read_metadata(metadata_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('*** RUNNING ON : ',device)
    embedding_model = vggish().to(device)
    embedding_model.eval()

    embedding_model.pproc._pca_matrix = embedding_model.pproc._pca_matrix.to(device)
    embedding_model.pproc._pca_means = embedding_model.pproc._pca_means.to(device)
    
    feats = []

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f"{len(relevant_files)} files found in metadata")

    for fi in tqdm(sorted(os.listdir(indir))):
        if fi in relevant_files:
            example = vggish_input.wavfile_to_examples(os.path.join(indir, fi))
            if example.shape[0] == 0:
               print(f"extraction in {fi}, skipping this file")
               continue
            example = example.to(device)
            with torch.no_grad():
                embeddings = embedding_model.forward(example).to(device)
                if embeddings.ndim == 1 and embeddings.shape[0] == 128:
                   embeddings = embeddings.unsqueeze(0)
                   
                #print(embeddings.shape)
                embeddings = torch.mean(embeddings, dim=0)
                #print(embeddings.shape)
                feats.append(embeddings.cpu().detach().numpy())

    if feats:
        feats = np.vstack(feats)
        np.save(os.path.join(outdir, 'dev_VGGish_27k.npy'), feats)


if __name__ == '__main__':
#    indir = 'flac_T/'
 #   outdir = '/Users/davidcombei/PycharmProjects/ASVSpoof5/saved_models'
  #  metadata_file = 'ASVspoof5.train_MEDIUM.metadata.txt'

    indir = '/home/asvspoof/DATA/asvspoof24/flac_D/'
    outdir = '/home/asvspoof/DATA/asvspoof24/VGGish'
    metadata_file = '/home/asvspoof/DATA/asvspoof24/metadata/ASVspoof5.dev_MEDIUM.metadata.txt'

    main(indir, outdir, metadata_file)
