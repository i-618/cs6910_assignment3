import torch, numpy as np

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

    def create_char_vocab_from_words(self, words):
        unique_chars = set()
        [unique_chars.update(list(x)) for x in words]
        unique_chars = list(unique_chars)
        unique_chars.sort()
        self.create_vocab(unique_chars)
    
    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.vocab_size
            self.inx2char[self.vocab_size] = char
            self.vocab_size += 1
        else:
            print("Character already exists in the script")
    

    def __len__(self):
        return self.vocab_size
    
    def __str__(self):
        return f"Script: {self.script_name}, Vocab Size: {self.vocab_size}"

def load_dataset_csv(path, START='<', END='>'):
    X, y = [], []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split(',')
            X.append(f'{START}{line[0]}{END}')
            y.append(f'{START}{line[1]}{END}')
    
    return X, y

def get_dataloader(transliter_pairs, latin_script, local_script, max_length, device, batch_size=32):
    n = len(transliter_pairs)
    input_ids = np.zeros((n, max_length), dtype=int)
    output_ids = np.zeros((n, max_length), dtype=int)


    for idx, (latin, local) in enumerate(transliter_pairs):
        inp_ids = [latin_script.char2idx[c] for c in latin]
        out_ids = [local_script.char2idx[c] for c in local]
        input_ids[idx, :len(inp_ids)] = inp_ids
        output_ids[idx, :len(out_ids)] = out_ids

    
    

    dataset = torch.utils.data.TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(output_ids).to(device))
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader