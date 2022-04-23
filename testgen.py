import IPython
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
import tensorflow as tf
from text import symbols

from hparams import create_hparams
from train import load_model
from text import text_to_sequence

import numpy as np

torch.random.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)

hifi = torch.jit.load("checkpoints/hifi_biden_fp16.jit")
c = torch.load("checkpoints/tacotron2_en_ms.pt")
del c['state_dict']['speaker_embedding.weight']
hparams = create_hparams()
model = load_model(hparams)
model.load_state_dict(c['state_dict'])
_ = model.cuda().eval().half()
text = "I love ice  cream"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval().half()

for k in waveglow.convinv:
    k.float()
audio = None
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
torchaudio.save("output_waveglow.wav", audio[0:1].cpu(), rate=hparams.sampling_rate)

