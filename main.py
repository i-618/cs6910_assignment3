import argparse
import wandb
from langscript import Script, load_dataset_csv, get_dataloader
from sequenceModel import AttnDecoderRNN, Encoder, Decoder, train
import torch
import os



# wandb.login(key = 'WANDB_API_KEY', verify = True)

run = wandb.init(
      # Set the project where this run will be logged
      project="CS6910-Assignment-3"
      )

parser = argparse.ArgumentParser(description='Sequential Network for Transliteration')

parser.add_argument('-wp', '--wandb_project', type=str, default='CS6910-Assignment-3', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', type=str, default='aksharantar_sampled', choices=["aksharantar_sampled"], help='Dataset choices: ["aksharantar_sampled"]')
parser.add_argument('-l', '--language', type=str, default='kan', help='Language choices: ["kan", "mal", "hin"]')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network.')
parser.add_argument('-hs', '--hidden_size', type=int, default=128, help='Hidden size of the Encoder and Decoder.')
parser.add_argument('-ie', '--input_embedding_size', type=int, default=128, help='Input embedding size of the Encoder.')
parser.add_argument('-nel', '--num_encoder_layers', type=int, default=2, help='Number of layers in the Encoder.')
parser.add_argument('-edp', '--encoder_dropout_p', type=float, default=0.1, help='Dropout probability in the Encoder.')
parser.add_argument('-ndl', '--num_decoder_layers', type=int, default=2, help='Number of layers in the Decoder.')
parser.add_argument('-ddp', '--decoder_dropout_p', type=float, default=0.1, help='Dropout probability in the Decoder.')
parser.add_argument('-bi', '--bidirectional', type=bool, default=False, help='Bidirectional Encoder.')
parser.add_argument('-ct', '--cell_type', type=str, default='RNN', help='Cell type used in Encoder and Decoder. Choices: ["GRU", "RNN"]')
parser.add_argument('-do', '--dropout', type=float, default=0.1, help='Dropout probability in the Encoder and Decoder.')
parser.add_argument('-attn', '--attention', type=bool, default=False, help='Use Attention in Decoder.')

args = parser.parse_args()






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)


# Load the dataset contraining words transliterated from one scipt to anothe

dataset_name = args.dataset
language = args.language
list_files = os.listdir(f'{dataset_name}/{language}')
path = f'{dataset_name}/{language}'


# Load the there data files 
X_test, y_test = load_dataset_csv(f'{path}/{language}_test.csv')
X_train, y_train = load_dataset_csv(f'{path}/{language}_train.csv')
X_val, y_val = load_dataset_csv(f'{path}/{language}_val.csv')

# MAX Length decides how many times the decoder will run and generate text, this is also used for padding
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

batch_size=args.batch_size
dataloader_train = get_dataloader(transliter_pairs_train, latin_script, local_script, MAX_LENGTH, device,batch_size=batch_size)
dataloader_test = get_dataloader(transliter_pairs_test, latin_script, local_script, MAX_LENGTH, device, batch_size=batch_size)
dataloader_val = get_dataloader(transliter_pairs_val, latin_script, local_script, MAX_LENGTH, device, batch_size=batch_size)

epochs = args.epochs
hidden_size = args.hidden_size
num_encoder_layers = args.num_encoder_layers
encoder_dropout_p = args.encoder_dropout_p
num_decoder_layers = args.num_decoder_layers
decoder_dropout_p = args.decoder_dropout_p
rnn_type = args.cell_type
bidirectional = args.bidirectional
RNN = getattr(torch.nn, rnn_type)
inp_embed_size = args.input_embedding_size
attention = args.attention
print('attention', attention)

if not args.wandb_sweepid is None:
    # For sweep runs, where config is set by wandb.ai
    def hyper_config_run(config=None):
        with wandb.init(config=config, reinit=True) as run:  # type: ignore
             # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            
            run_name = str(config).replace("': '", ' ').replace("'", '')
            print(run_name)
            run.name = run_name
            if attention:
                num_encoder_layers,num_decoder_layers  = 1, 1
                bidirectional = False
                input_embedding_size = config.input_embedding_size
                hidden_size = config.hidden_layer_size
                dropout_encoder = config.dropout_encoder
                dropout_decoder = config.dropout_decoder
                RNN_ENC = getattr(torch.nn, config.cell_type_encoder)
                RNN_DEC = getattr(torch.nn, config.cell_type_decoder)
                encoder = Encoder(input_size=latin_script.vocab_size, hidden_size=hidden_size, RNN=RNN_ENC,inp_emmbed_size=input_embedding_size, num_layers=num_encoder_layers, bidirectional=False, dropout_p=dropout_encoder).to(device)
                decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=local_script.vocab_size, max_length=MAX_LENGTH, RNN=RNN_DEC, dropout_p=dropout_decoder, device=device).to(device)

            else:
                
                hidden_size = config.hidden_layer_size
                num_encoder_layers = config.num_encoder_layers
                encoder_dropout_p = config.dropout
                num_decoder_layers = config.num_decoder_layers
                decoder_dropout_p = config.dropout
                bidirectional = True if config.bidirectional == 'Yes' else False
                inp_embed_size = config.input_embedding_size
                RNN = getattr(torch.nn, config.cell_type)
                encoder = Encoder(input_size=latin_script.vocab_size, hidden_size=hidden_size, RNN=RNN, inp_emmbed_size=inp_embed_size,num_layers=num_encoder_layers, bidirectional=bidirectional, dropout_p=encoder_dropout_p).to(device)
                decoder = Decoder(hidden_size=hidden_size, max_length=MAX_LENGTH, RNN=RNN, output_size=local_script.vocab_size, dropout_p=decoder_dropout_p, bidirectional=bidirectional, num_decoder_layers=num_decoder_layers, device=device).to(device)
            train(dataloader_train, encoder, decoder, epochs, print_every=1, device=device, val_dataloader=dataloader_val)
        
    wandb.agent(args.wandb_sweepid, project=args.wandb_project, function=hyper_config_run)
else:
    if attention:
        # Bahadanau Attention was not working with multiple layers
        num_encoder_layers,num_decoder_layers  = 1, 1
        bidirectional = False
        dropout_encoder = encoder_dropout_p
        dropout_decoder = decoder_dropout_p
        RNN_ENC = RNN
        RNN_DEC = getattr(torch.nn, 'GRU')
        encoder = Encoder(input_size=latin_script.vocab_size, hidden_size=hidden_size, RNN=RNN_ENC,inp_emmbed_size=inp_embed_size, num_layers=num_encoder_layers, bidirectional=False, dropout_p=dropout_encoder).to(device)
        decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=local_script.vocab_size, max_length=MAX_LENGTH, RNN=RNN_DEC, dropout_p=dropout_decoder, device=device).to(device)
    else:
        encoder = Encoder(input_size=latin_script.vocab_size, hidden_size=hidden_size, inp_emmbed_size=inp_embed_size, num_layers=num_encoder_layers, bidirectional=bidirectional, dropout_p=encoder_dropout_p).to(device)
        decoder = Decoder(hidden_size=hidden_size, max_length=MAX_LENGTH, output_size=local_script.vocab_size, bidirectional=bidirectional, dropout_p=decoder_dropout_p, num_decoder_layers=num_decoder_layers, device=device).to(device)
    loss = train(dataloader_train, encoder, decoder, epochs, print_every=1, device=device, val_dataloader=dataloader_val)

