import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer_whisper import LightningModel
from dataset import S2IMELDataset, collate_fn

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

dataset = S2IMELDataset(
        csv_path="/root/Speech2Intent/dataset/speech-to-intent/test.csv",
        wav_dir_path="/root/Speech2Intent/dataset/speech-to-intent/",
    )

# change path to checkpoint
model = LightningModel.load_from_checkpoint("/root/Speech2Intent/s2i-baselines/checkpoints/whisper_asr_small.ckpt")
model.to('cuda')
model.eval()

trues=[]
preds = []

for x, label in tqdm(dataset):
    x_tensor = x.to("cuda").unsqueeze(0)
    y_hat_l = model(x_tensor)

    probs = F.softmax(y_hat_l, dim=1).detach().cpu().view(1, 14)
    pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
    probs = probs.numpy().astype(float).tolist()
    trues.append(label)
    preds.append(pred)
print(f"Accuracy Score = {accuracy_score(trues, preds)}\nF1-Score = {f1_score(trues, preds, average='weighted')}")