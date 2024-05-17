import torch, numpy as np

class Script:
    """
    Represents a script with character-to-index and index-to-character mappings.

    Attributes:
        script_name (str): The name of the script.
        char2idx (dict): A dictionary mapping characters to their corresponding indices.
        inx2char (dict): A dictionary mapping indices to their corresponding characters.
        vocab_size (int): The size of the vocabulary.

    Methods:
        create_vocab(char_list): Creates the vocabulary from a list of characters.
        create_char_vocab_from_words(words): Creates the vocabulary from a list of words.
        add_char(char): Adds a character to the vocabulary.
        __len__(): Returns the size of the vocabulary.
        __str__(): Returns a string representation of the Script object.
    """

    def __init__(self, script_name):
        self.script_name = script_name
        self.char2idx = {}
        self.inx2char = {}
        self.vocab_size = 0

    def create_vocab(self, char_list):
        """
        Creates the vocabulary from a list of characters.

        Args:
            char_list (list): A list of characters.

        Returns:
            None
        """
        for i, char in enumerate(char_list):
            self.char2idx[char] = i
            self.inx2char[i] = char
        self.vocab_size = len(char_list)

    def create_char_vocab_from_words(self, words):
        """
        Creates the vocabulary from a list of words.

        Args:
            words (list): A list of words.

        Returns:
            None
        """
        unique_chars = set()
        [unique_chars.update(list(x)) for x in words]
        unique_chars = list(unique_chars)
        unique_chars.sort()
        self.create_vocab(unique_chars)
    
    def add_char(self, char):
        """
        Adds a character to the vocabulary.

        Args:
            char (str): The character to be added.

        Returns:
            None
        """
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
    """
    Load a dataset from a CSV file.

    Args:
        path (str): The path to the CSV file.
        START (str, optional): The start token to prepend to each line. Defaults to '<'.
        END (str, optional): The end token to append to each line. Defaults to '>'.

    Returns:
        tuple: A tuple containing two lists - X and y.
            X (list): A list of input lines with start and end tokens.
            y (list): A list of output lines with start and end tokens.
    """
    X, y = [], []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split(',')
            X.append(f'{START}{line[0]}{END}')
            y.append(f'{START}{line[1]}{END}')
    return X, y

def get_dataloader(transliter_pairs, latin_script, local_script, max_length, device, batch_size=32):
    """
    Create and return a PyTorch DataLoader for the given transliteration pairs.

    Args:
        transliter_pairs (list): A list of tuples representing the transliteration pairs.
        latin_script (Script): The script object for the Latin script.
        local_script (Script): The script object for the local script.
        max_length (int): The maximum length of the input and output sequences.
        device (torch.device): The device to use for the DataLoader.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader object containing the input and output sequences.

    """
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