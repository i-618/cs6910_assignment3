import math
import torchmetrics
from langscript import Script, load_dataset_csv, get_dataloader
from sequenceModel import AttnDecoderRNN, Encoder, Decoder, train
import torch
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset contraining words transliterated from one scipt to anothe

dataset_name = "aksharantar_sampled"
language = "kan"
list_files = os.listdir(f'{dataset_name}/{language}')
path = f'{dataset_name}/{language}'



X_test, y_test = load_dataset_csv(f'{path}/{list_files[0]}')
X_train, y_train = load_dataset_csv(f'{path}/{list_files[1]}')
X_val, y_val = load_dataset_csv(f'{path}/{list_files[2]}')

MAX_LENGTH = max([len(x) for x in X_train] + [len(y) for y in y_train])

print('MAX_LENGTH', MAX_LENGTH)
# Create the script object
local_script = Script(language)
local_script.create_char_vocab_from_words(y_train)
print(local_script)

latin_script = Script("latin")
latin_script.create_char_vocab_from_words(X_train)
print(latin_script)

transliter_pairs_test = list(zip(X_test, y_test))
transliter_pairs_train = list(zip(X_train, y_train))
transliter_pairs_val = list(zip(X_val, y_val))

dataloader_train = get_dataloader(transliter_pairs_train, latin_script, local_script, MAX_LENGTH, device,batch_size=32)
dataloader_test = get_dataloader(transliter_pairs_test, latin_script, local_script, MAX_LENGTH, device, batch_size=32)
dataloader_val = get_dataloader(transliter_pairs_val, latin_script, local_script, MAX_LENGTH, device, batch_size=32)

hidden_size = 128
batch_size=32
encoder = Encoder(input_size=latin_script.vocab_size, hidden_size=hidden_size, num_layers=5,dropout_p=0).to(device)
decoder = Decoder(hidden_size=hidden_size, max_length=MAX_LENGTH, output_size=local_script.vocab_size, num_decoder_layers=2,device=device).to(device)

attn_decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=local_script.vocab_size, device=device).to(device)

loss = train(dataloader_train, encoder, decoder, 25, print_every=1, plot_every=1, device=device, val_dataloader=dataloader_val)
