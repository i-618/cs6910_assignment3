# cs6910_assignment3
Assignment Solved by Likhith Kumara (NS24Z066) for the course CS6910 Fundamentals of Deep Learning

This repoistory has the code for a sequence to sequence model for transliteration from one script to another. The dataset used is aksharantar. The code for hyperparameter tuning is also included.

There are three important files:
1. main.py: the file that imports the other two files to build the model and train it.
2. sequenceModel.py: the file that contains the model class and the training and evaluation functions.
3. langscript.py: the file that contains the class for the language and script, mainly for reading the data and storing them.

The code is runnable via command line arguments:
```
usage: main.py [-h] [-wp WANDB_PROJECT] [-sid WANDB_SWEEPID] [-d {aksharantar_sampled}] [-l LANGUAGE] [-e EPOCHS] [-b BATCH_SIZE] [-hs HIDDEN_SIZE]
               [-ie INPUT_EMBEDDING_SIZE] [-nel NUM_ENCODER_LAYERS] [-edp ENCODER_DROPOUT_P] [-ndl NUM_DECODER_LAYERS] [-ddp DECODER_DROPOUT_P]
               [-bi BIDIRECTIONAL] [-ct CELL_TYPE] [-do DROPOUT] [-attn ATTENTION]

Sequential Network for Transliteration

optional arguments:
  -h, --help            show this help message and exit
  -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT
                        Project name used to track experiments in Weights & Biases dashboard
  -sid WANDB_SWEEPID, --wandb_sweepid WANDB_SWEEPID
                        Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.
  -d {aksharantar_sampled}, --dataset {aksharantar_sampled}
                        Dataset choices: ["aksharantar_sampled"]
  -l LANGUAGE, --language LANGUAGE
                        Language choices: ["kan", "mal", "hin"]
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train neural network.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size used to train neural network.
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Hidden size of the Encoder and Decoder.
  -ie INPUT_EMBEDDING_SIZE, --input_embedding_size INPUT_EMBEDDING_SIZE
                        Input embedding size of the Encoder.
  -nel NUM_ENCODER_LAYERS, --num_encoder_layers NUM_ENCODER_LAYERS
                        Number of layers in the Encoder.
  -edp ENCODER_DROPOUT_P, --encoder_dropout_p ENCODER_DROPOUT_P
                        Dropout probability in the Encoder.
  -ndl NUM_DECODER_LAYERS, --num_decoder_layers NUM_DECODER_LAYERS
                        Number of layers in the Decoder.
  -ddp DECODER_DROPOUT_P, --decoder_dropout_p DECODER_DROPOUT_P
                        Dropout probability in the Decoder.
  -bi BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
                        Bidirectional Encoder.
  -ct CELL_TYPE, --cell_type CELL_TYPE
                        Cell type used in Encoder and Decoder. Choices: ["GRU", "RNN"]
  -do DROPOUT, --dropout DROPOUT
                        Dropout probability in the Encoder and Decoder.
  -attn ATTENTION, --attention ATTENTION
                        Use Attention in Decoder.
```