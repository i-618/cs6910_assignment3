import math
import torchmetrics
from languageModel import LanguageModel
import torch
import pytorch_lightning as pl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset contraining words transliterated from one scipt to another

# class to store the alphabets and their corresponding indices
class Script:
    def __init__(self, script_name):
        self.script_name = script_name
        self.char2idx = {}
        self.inx2char = {}
        self.vocab_size = 0

    def create_vocab(self, char_list):
        for i, char in enumerate(char_list):
            self.char2idx[char] = i
            self.inx2char[i] = char
        self.vocab_size = len(char_list)
    
    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.vocab_size
            self.inx2char[self.vocab_size] = char
            self.vocab_size += 1
        else:
            print("Character already exists in the script")




    def __len__(self):
        return len(self.script)

    def __getitem__(self, item):
        return self.script[item]




# optimizer_function = torch.optim.Adam
# optimizer_params = {}
# accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
# loss_function = torch.nn.CrossEntropyLoss()

# model = LanguageModel(model=model, loss_function=loss_function, accuracy_function=accuracy_function, 
#                         optimizer_function=optimizer_function, optimizer_params=optimizer_params)

# # model.to(device)
# print(model)
# trainer  = pl.Trainer(log_every_n_steps=5, max_epochs=100)
# train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=int(len(train_data)/3))
# val_dataloaders = torch.utils.data.DataLoader(val_data)

# trainer.fit( model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)





