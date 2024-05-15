import torch
import time, torchmetrics, math

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2,dropout_p=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=num_layers,batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length=32, num_decoder_layers=3,device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.num_decoder_layers = num_decoder_layers
        self.max_length = max_length
        self.device = device
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=num_decoder_layers, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)

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

    def forward_step(self, input:torch.Tensor, hidden:torch.Tensor):
        output = self.embedding(input)
        output = torch.nn.functional.relu(output)
        if hidden.shape[0] != self.num_decoder_layers:
            hidden = hidden.repeat(self.num_decoder_layers, 1, 1)
        output, hidden = self.gru(output, hidden[-self.num_decoder_layers:])
        output = self.out(output)
        return output, hidden
    
class BahdanauAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = torch.nn.Linear(hidden_size, hidden_size)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size)
        self.Va = torch.nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, max_length=29, dropout_p=0.1, num_decoder_layers=3, device=torch.device('cpu')):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = torch.nn.GRU(2 * hidden_size, hidden_size, num_layers=num_decoder_layers, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.max_length = max_length
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
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


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
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
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
    return decoder_outputs

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100, device=torch.device('cpu')
               ,val_dataloader=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

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
            val_accuracy = calclulate_validation_accuracy(val_dataloader, accuracy_criterion, encoder, decoder, encode_decode_inp, device=device)
            print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100:.2f}%) Loss: {print_loss_avg:.4f} \
                  Acc: {print_accuracy_avg*100:.2f} %, Val Acc: {val_accuracy*100:.2f} %')

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    
    return plot_losses

def calclulate_validation_accuracy(val_dataloader, accuracy_criterion, encoder, decoder, encode_decode_inp, device=torch.device('cpu')):
    total_accuracy = torch.tensor([], dtype=torch.float32, device=device)
    for data in val_dataloader:
        input_tensor, target_tensor = data
        decoder_outputs = encode_decode_inp(encoder, decoder, input_tensor, target_tensor)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        accuracy = accuracy_criterion(decoded_ids, target_tensor)
        total_accuracy = torch.cat((total_accuracy, accuracy))
    return sum(total_accuracy ==1)/len(total_accuracy)