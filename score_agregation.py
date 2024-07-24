import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

###########
###wavLM-V3
wavLM_V3_00 = np.load('eval_BIG_feats_wavLM-V3_00.npy', allow_pickle=True).item()
wavLM_V3_01 = np.load('eval_BIG_feats_wavLM-V3_01.npy', allow_pickle=True).item()
wavLM_V3_02 = np.load('eval_BIG_feats_wavLM-V3_02.npy', allow_pickle=True).item()
wavLM_V3_03 = np.load('eval_BIG_feats_wavLM-V3_03.npy', allow_pickle=True).item()
wavLM_V3_04 = np.load('eval_BIG_feats_wavLM-V3_04.npy', allow_pickle=True).item()
wavLM_V3_05 = np.load('eval_BIG_feats_wavLM-V3_05.npy', allow_pickle=True).item()
wavLM_V3_06 = np.load('eval_BIG_feats_wavLM-V3_06.npy', allow_pickle=True).item()

files0_V3 = wavLM_V3_00.get('files')
files1_V3 = wavLM_V3_01.get('files')
files2_V3 = wavLM_V3_02.get('files')
files3_V3 = wavLM_V3_03.get('files')
files4_V3 = wavLM_V3_04.get('files')
files5_V3 = wavLM_V3_05.get('files')
files6_V3 = wavLM_V3_06.get('files')

feats0_V3 = wavLM_V3_00.get('feats')
feats1_V3 = wavLM_V3_01.get('feats')
feats2_V3 = wavLM_V3_02.get('feats')
feats3_V3 = wavLM_V3_03.get('feats')
feats4_V3 = wavLM_V3_04.get('feats')
feats5_V3 = wavLM_V3_05.get('feats')
feats6_V3 = wavLM_V3_06.get('feats')


###########
###wavLM-V2
wavLM_V2_00 = np.load('eval_BIG_feats_wavLM-UTCN_00.npy', allow_pickle=True).item()
wavLM_V2_01 = np.load('eval_BIG_feats_wavLM-UTCN_01.npy', allow_pickle=True).item()
wavLM_V2_02 = np.load('eval_BIG_feats_wavLM-UTCN_02.npy', allow_pickle=True).item()
wavLM_V2_03 = np.load('eval_BIG_feats_wavLM-UTCN_03.npy', allow_pickle=True).item()
wavLM_V2_04 = np.load('eval_BIG_feats_wavLM-UTCN_04.npy', allow_pickle=True).item()
wavLM_V2_05 = np.load('eval_BIG_feats_wavLM-UTCN_05.npy', allow_pickle=True).item()
wavLM_V2_06 = np.load('eval_BIG_feats_wavLM-UTCN_06.npy', allow_pickle=True).item()

files0_V2 = wavLM_V2_00.get('files')
files1_V2 = wavLM_V2_01.get('files')
files2_V2 = wavLM_V2_02.get('files')
files3_V2 = wavLM_V2_03.get('files')
files4_V2 = wavLM_V2_04.get('files')
files5_V2 = wavLM_V2_05.get('files')
files6_V2 = wavLM_V2_06.get('files')

feats0_V2 = wavLM_V2_00.get('feats')
feats1_V2 = wavLM_V2_01.get('feats')
feats2_V2 = wavLM_V2_02.get('feats')
feats3_V2 = wavLM_V2_03.get('feats')
feats4_V2 = wavLM_V2_04.get('feats')
feats5_V2 = wavLM_V2_05.get('feats')
feats6_V2 = wavLM_V2_06.get('feats')



###########
###wavLM-V1
wavLM_V1_00 = np.load('eval_BIG_feats_wavLM-base-finetuned_00.npy', allow_pickle=True).item()
wavLM_V1_01 = np.load('eval_BIG_feats_wavLM-base-finetuned_01.npy', allow_pickle=True).item()
wavLM_V1_02 = np.load('eval_BIG_feats_wavLM-base-finetuned_02.npy', allow_pickle=True).item()
wavLM_V1_03 = np.load('eval_BIG_feats_wavLM-base-finetuned_03.npy', allow_pickle=True).item()
wavLM_V1_04 = np.load('eval_BIG_feats_wavLM-base-finetuned_04.npy', allow_pickle=True).item()
wavLM_V1_05 = np.load('eval_BIG_feats_wavLM-base-finetuned_05.npy', allow_pickle=True).item()
wavLM_V1_06 = np.load('eval_BIG_feats_wavLM-base-finetuned_06.npy', allow_pickle=True).item()

files0_V1 = wavLM_V1_00.get('files')
files1_V1 = wavLM_V1_01.get('files')
files2_V1 = wavLM_V1_02.get('files')
files3_V1 = wavLM_V1_03.get('files')
files4_V1 = wavLM_V1_04.get('files')
files5_V1 = wavLM_V1_05.get('files')
files6_V1 = wavLM_V1_06.get('files')

feats0_V1 = wavLM_V1_00.get('feats')
feats1_V1 = wavLM_V1_01.get('feats')
feats2_V1 = wavLM_V1_02.get('feats')
feats3_V1 = wavLM_V1_03.get('feats')
feats4_V1 = wavLM_V1_04.get('feats')
feats5_V1 = wavLM_V1_05.get('feats')
feats6_V1 = wavLM_V1_06.get('feats')


###########
###wavLM-base
wavLM_base_00 = np.load('eval_BIG_feats_wavlm-base_00.npy', allow_pickle=True).item()
wavLM_base_01 = np.load('eval_BIG_feats_wavlm-base_01.npy', allow_pickle=True).item()
wavLM_base_02 = np.load('eval_BIG_feats_wavlm-base_02.npy', allow_pickle=True).item()
wavLM_base_03 = np.load('eval_BIG_feats_wavlm-base_03.npy', allow_pickle=True).item()
wavLM_base_04 = np.load('eval_BIG_feats_wavlm-base_04.npy', allow_pickle=True).item()
wavLM_base_05 = np.load('eval_BIG_feats_wavlm-base_05.npy', allow_pickle=True).item()
wavLM_base_06 = np.load('eval_BIG_feats_wavlm-base_06.npy', allow_pickle=True).item()

files0_base = wavLM_base_00.get('files')
files1_base = wavLM_base_01.get('files')
files2_base = wavLM_base_02.get('files')
files3_base = wavLM_base_03.get('files')
files4_base = wavLM_base_04.get('files')
files5_base = wavLM_base_05.get('files')
files6_base = wavLM_base_06.get('files')

feats0_base = wavLM_base_00.get('feats')
feats1_base = wavLM_base_01.get('feats')
feats2_base = wavLM_base_02.get('feats')
feats3_base = wavLM_base_03.get('feats')
feats4_base = wavLM_base_04.get('feats')
feats5_base = wavLM_base_05.get('feats')
feats6_base = wavLM_base_06.get('feats')






train1 = np.load('train_feats_27k.npy')
train2 = np.load('train_feats_finetuned_27k.npy')
train3 = np.load('train_feats_finetunedV2_27k.npy')
train4 = np.load('train_feats_finetunedV3_27k.npy')

train_labels = np.load('train_labels_1output_27k.npy')

dev1 = np.load('dev_feats_27k.npy')
dev2 = np.load('dev_feats_finetuned_27k.npy')
dev3 = np.load('dev_feats_finetunedV2_27k.npy')
dev4 = np.load('dev_feats_finetunedV3_27k.npy')
dev_labels = np.load('dev_labels_1output_27k.npy')

eval1 = np.concatenate((feats0_V3,feats1_V3,feats2_V3,feats3_V3,feats4_V3,feats5_V3,feats6_V3), axis=0)
eval2 = np.concatenate((feats0_V2,feats1_V2,feats2_V2,feats3_V2,feats4_V2,feats5_V2,feats6_V2), axis=0)
eval3 = np.concatenate((feats0_V1,feats1_V1,feats2_V1,feats3_V1,feats4_V1,feats5_V1,feats6_V1),axis=0)
eval4 = np.concatenate((feats0_base,feats1_base,feats2_base,feats3_base,feats4_base,feats5_base,feats6_base),axis=0)
print(eval1.shape)
print(eval2.shape)
print(eval3.shape)
print(eval4.shape)


model1 = LogisticRegression(random_state=46, C=1e3, max_iter=10000)
model1.fit(train1, train_labels)

model2 = LogisticRegression(random_state=46, C=1e3, max_iter=10000)
model2.fit(train2, train_labels)

model3 = LogisticRegression(random_state=46, C=1e3, max_iter=10000)
model3.fit(train3, train_labels)

model4 = LogisticRegression(random_state=46, C=1e3, max_iter=10000)
model4.fit(train4, train_labels)



train_prob1 = model1.predict_proba(train1)[:, 1].reshape(-1, 1)
train_prob2 = model2.predict_proba(train2)[:, 1].reshape(-1, 1)
train_prob3 = model3.predict_proba(train3)[:, 1].reshape(-1, 1)
train_prob4 = model4.predict_proba(train4)[:, 1].reshape(-1, 1)


train_combined_probs = np.hstack((train_prob1, train_prob2, train_prob3, train_prob4))


final_model = LogisticRegression(random_state=46, C=1e3, max_iter=10000)
final_model.fit(train_combined_probs, train_labels)

val_prob1 = model1.predict_proba(dev1)[:, 1].reshape(-1, 1)
val_prob2 = model2.predict_proba(dev2)[:, 1].reshape(-1, 1)
val_prob3 = model3.predict_proba(dev3)[:, 1].reshape(-1, 1)
val_prob4 = model4.predict_proba(dev4)[:, 1].reshape(-1, 1)


val_combined_probs = np.hstack((val_prob1, val_prob2, val_prob3, val_prob4))


print("*** Evaluating model...")
y_val_pred = final_model.predict(val_combined_probs)
y_val_prob = final_model.predict_proba(val_combined_probs)[:, 1]

fpr, tpr, thresholds = roc_curve(dev_labels, y_val_prob, pos_label=1)
fnr = 1 - tpr
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print(f"Validation EER: {eer * 100:.2f}")

final_model.fit(val_combined_probs, dev_labels)

eval_prob1 = model1.predict_proba(eval1)[:, 1].reshape(-1, 1)
eval_prob2 = model2.predict_proba(eval2)[:, 1].reshape(-1, 1)
eval_prob3 = model3.predict_proba(eval3)[:, 1].reshape(-1, 1)
eval_prob4 = model4.predict_proba(eval4)[:, 1].reshape(-1, 1)


eval_combined_probs = np.hstack((eval_prob1, eval_prob2, eval_prob3, eval_prob4))
#
#
file_names = files0_V3+files1_V3+files2_V3+files3_V3+files4_V3+files5_V3+files6_V3
final_prob = final_model.predict_proba(eval_combined_probs)[:, 1]


output_file = '/home/asvspoof/DATA/asvspoof24/wavLM-base/score.tsv'

with open(output_file, 'w') as file:
    file.write('filename\tcm-score\n')
    for s, value in zip(file_names, final_prob):
        file.write(f'{s}\t{value:.4f}\n')

print('TSV file created! good luck')

