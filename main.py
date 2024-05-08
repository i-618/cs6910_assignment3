import math
import torchmetrics
from langscript import Script
from sequenceModel import EncoderDecoder, SelectItem
import torch
import pytorch_lightning as pl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset contraining words transliterated from one scipt to anothe




