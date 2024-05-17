import time, torch, torchmetrics, math, wandb

class Encoder(torch.nn.Module):
    """
    Encoder module that takes input sequences and produces hidden states.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers. Default is 2.
        inp_emmbed_size (int, optional): The size of the input embedding. Default is 16.
        RNN (torch.nn.Module, optional): The RNN module to use. Default is torch.nn.GRU.
        dropout_p (float, optional): The probability of dropout. Default is 0.1.
        bidirectional (bool, optional): If True, becomes a bidirectional encoder. Default is False.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, inp_emmbed_size=16, RNN=torch.nn.GRU, dropout_p=0.1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, inp_emmbed_size)
        self.gru = RNN(hidden_size=hidden_size, input_size=inp_emmbed_size, num_layers=num_layers,batch_first=True, \
                        dropout=dropout_p if num_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class Decoder(torch.nn.Module):
    """
    Decoder module for sequence-to-sequence models.

    Args:
        hidden_size (int): The number of features in the hidden state of the RNN.
        output_size (int): The number of output classes.
        max_length (int, optional): The maximum length of the output sequence. Defaults to 32.
        num_decoder_layers (int, optional): The number of layers in the decoder RNN. Defaults to 3.
        RNN (torch.nn.Module, optional): The type of RNN to use. Defaults to torch.nn.GRU.
        dropout_p (float, optional): The probability of dropout. Defaults to 0.1.
        bidirectional (bool, optional): If True, the RNN layers are bidirectional. Defaults to False.
        device (torch.device, optional): The device to use for computation. Defaults to torch.device('cpu').

    Attributes:
        output_size (int): The number of output classes.
        num_decoder_layers (int): The number of layers in the decoder RNN.
        max_length (int): The maximum length of the output sequence.
        device (torch.device): The device to use for computation.
        embedding (torch.nn.Embedding): The embedding layer for the decoder input.
        rnn (torch.nn.Module): The RNN module.
        dropout (torch.nn.Dropout): The dropout layer.
        out (torch.nn.Linear): The linear layer for the output.

    """

    def __init__(self, hidden_size, output_size, max_length=32, num_decoder_layers=3, RNN=torch.nn.LSTM, dropout_p=0.1, bidirectional=False,  device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.num_decoder_layers = num_decoder_layers
        self.max_length = max_length
        self.device = device
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.rnn = RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_decoder_layers, batch_first=True, \
                       dropout=dropout_p if num_decoder_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout_p)
 
        self.out = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input_tensor:torch.Tensor, hidden:torch.Tensor):
        output = self.dropout(self.embedding(input_tensor))
        output = torch.nn.functional.relu(output)
        if hidden.shape[0] != self.num_decoder_layers * (2 if self.rnn.bidirectional else 1):
            hidden = hidden.repeat(self.num_decoder_layers * (2 if self.rnn.bidirectional else 1), 1, 1)
        output, hidden = self.rnn(output, hidden[-self.num_decoder_layers * (2 if self.rnn.bidirectional else 1):])
        output = self.out(output)
        return output, hidden
    
class BahdanauAttention(torch.nn.Module):
    """
    Bahdanau Attention mechanism for sequence models.

    Args:
        hidden_size (int): The size of the hidden state.

    Attributes:
        Wa (torch.nn.Linear): Linear layer for transforming the query.
        Ua (torch.nn.Linear): Linear layer for transforming the keys.
        Va (torch.nn.Linear): Linear layer for calculating the attention scores.

    Methods:
        forward(query, keys): Performs the forward pass of the attention mechanism.

    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = torch.nn.Linear(hidden_size, hidden_size)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size)
        self.Va = torch.nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        """
        Performs the forward pass of the attention mechanism.

        Args:
            query (torch.Tensor): The query tensor.
            keys (torch.Tensor): The keys tensor.

        Returns:
            context (torch.Tensor): The context vector.
            weights (torch.Tensor): The attention weights.

        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(torch.nn.Module):
    """
    Attention-based Decoder RNN for sequence-to-sequence models.

    Args:
        hidden_size (int): The size of the hidden state of the decoder.
        output_size (int): The size of the output vocabulary.
        max_length (int, optional): The maximum length of the output sequence. Defaults to 29.
        RNN (torch.nn.Module, optional): The RNN module to use for the decoder. Defaults to torch.nn.GRU.
        dropout_p (float, optional): The dropout probability. Defaults to 0.1.
        device (torch.device, optional): The device to use for computation. Defaults to torch.device('cpu').

    Attributes:
        output_size (int): The size of the output vocabulary.
        embedding (torch.nn.Embedding): The embedding layer for the decoder input.
        attention (BahdanauAttention): The attention mechanism.
        RNN (torch.nn.Module): The RNN module used for the decoder.
        out (torch.nn.Linear): The linear layer for the decoder output.
        dropout (torch.nn.Dropout): The dropout layer.
        max_length (int): The maximum length of the output sequence.
        device (torch.device): The device used for computation.

    Methods:
        forward(encoder_outputs, encoder_hidden, target_tensor=None):
            Performs the forward pass of the decoder.
        forward_step(input_tensor, hidden, encoder_outputs):
            Performs a single step of the decoder forward pass.

    """

    def __init__(self, hidden_size, output_size, max_length=29, RNN=torch.nn.GRU, dropout_p=0.1, device=torch.device('cpu')):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.RNN = RNN(2 * hidden_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.max_length = max_length
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """
        Performs the forward pass of the decoder.

        Args:
            encoder_outputs (torch.Tensor): The output sequence from the encoder. Shape: (batch_size, seq_length, hidden_size).
            encoder_hidden (torch.Tensor): The final hidden state of the encoder. Shape: (num_layers * num_directions, batch_size, hidden_size).
            target_tensor (torch.Tensor, optional): The target output sequence. Shape: (batch_size, seq_length). Defaults to None.

        Returns:
            decoder_outputs (torch.Tensor): The output sequence from the decoder. Shape: (batch_size, seq_length, output_size).
            decoder_hidden (torch.Tensor): The final hidden state of the decoder. Shape: (num_layers * num_directions, batch_size, hidden_size).
            attentions (torch.Tensor): The attention weights for each decoder step. Shape: (batch_size, seq_length, seq_length).

        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input_tensor, hidden, encoder_outputs):
        """
        Performs a single step of the decoder forward pass. This is used beause the lenght of the output is variable and for loop
        is used for generating each output.

        Args:
            input_tensor (torch.Tensor): The input tensor for the current step. Shape: (batch_size, 1).
            hidden (torch.Tensor): The hidden state of the decoder. Shape: (num_layers * num_directions, batch_size, hidden_size).
            encoder_outputs (torch.Tensor): The output sequence from the encoder. Shape: (batch_size, seq_length, hidden_size).

        Returns:
            output (torch.Tensor): The output tensor for the current step. Shape: (batch_size, 1, output_size).
            hidden (torch.Tensor): The updated hidden state of the decoder. Shape: (num_layers * num_directions, batch_size, hidden_size).
            attn_weights (torch.Tensor): The attention weights for the current step. Shape: (batch_size, 1, seq_length).

        """
        embedded =  self.dropout(self.embedding(input_tensor))
        if type(hidden) == tuple:
            hidden = hidden[0]
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.RNN(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, accuracy_criterion, device):
    """
    Trains the model for one epoch using the data that has been previously batched. So based on the batch size, the model is trained

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        encoder_optimizer (torch.optim.Optimizer): The optimizer for the encoder.
        decoder_optimizer (torch.optim.Optimizer): The optimizer for the decoder.
        criterion (torch.nn.Module): The loss function.
        accuracy_criterion (torch.nn.Module): The accuracy function.
        device (torch.device): The device to run the training on.

    Returns:
        float: The average loss for the epoch.
        torch.Tensor: The accuracy for each batch in the epoch.
    """

    total_loss = 0
    total_accuracy = torch.tensor([], dtype=torch.float32, device=device)
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        decoder_outputs = encode_decode_inp(encoder, decoder, input_tensor, target_tensor)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        accuracy = accuracy_criterion(decoded_ids, target_tensor)
        total_accuracy = torch.cat((total_accuracy, accuracy))
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), total_accuracy

def encode_decode_inp(encoder, decoder, input_tensor, target_tensor):
    """Common method to encode and decode the input tensor.

    Args:
        encoder (Model): torch model
        decoder (Model): torch model
        input_tensor (Tensor): torch tensor
        target_tensor (Tensor): torch tensor

    Returns:
        _type_: _description_
    """
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
    return decoder_outputs

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, device=torch.device('cpu')
               ,val_dataloader=None):
    start = time.time()
    
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Initialize the optimizer and criterion, Using Adam as it is the most popular optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()
    accuracy_criterion = torchmetrics.Accuracy(task='multiclass', num_classes=decoder.output_size, multidim_average='samplewise')


    for epoch in range(1, n_epochs + 1):
        loss, accuracy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, accuracy_criterion, device=device)
        print_loss_total += loss
        plot_loss_total += loss
        print_accuracy_total += sum(accuracy ==1)/len(accuracy)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_accuracy_avg = print_accuracy_total / print_every
            print_accuracy_total = 0
            print_loss_total = 0
            val_accuracy, char_wise_val_acc = calclulate_validation_accuracy(val_dataloader, accuracy_criterion, encoder, decoder, encode_decode_inp, device=device)
            print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100:.2f}%) Loss: {print_loss_avg:.4f}, \
Acc: {print_accuracy_avg*100:.2f} %, Val Acc: {val_accuracy*100:.2f} %, ChrValAcc: {char_wise_val_acc*100:.2f}')
            wandb.log({"train_loss": print_loss_avg, "train_accuracy": print_accuracy_avg, "val_accuracy": val_accuracy, "epoch": epoch, "char_wise_val_acc":char_wise_val_acc})
    return print_loss_total

def calclulate_validation_accuracy(val_dataloader, accuracy_criterion, encoder, decoder, encode_decode_inp, device=torch.device('cpu')):
    """
    Calculates the validation accuracy for a given validation dataloader.

    Args:
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        accuracy_criterion (torch.nn.Module): The accuracy criterion used for evaluation.
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        encode_decode_inp (callable): A function that takes the encoder, decoder, input tensor, and target tensor as input and returns the decoder outputs.
        device (torch.device, optional): The device to perform the computation on (default: torch.device('cpu')).

    Returns:
        Tuple[float, float]: A tuple containing the accuracy of completely correct predictions and the average accuracy across all predictions.
    """
    total_accuracy = torch.tensor([], dtype=torch.float32, device=device)
    for data in val_dataloader:
        input_tensor, target_tensor = data
        decoder_outputs = encode_decode_inp(encoder, decoder, input_tensor, target_tensor)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        accuracy = accuracy_criterion(decoded_ids, target_tensor)
        total_accuracy = torch.cat((total_accuracy, accuracy))
    return sum(total_accuracy == 1) / len(total_accuracy), sum(total_accuracy) / len(total_accuracy)
