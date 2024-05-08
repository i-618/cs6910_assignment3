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
        return self.vocab_size
    
    def __str__(self):
        return f"Script: {self.script_name}, Vocab Size: {self.vocab_size}"

