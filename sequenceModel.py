import torch
import pytorch_lightning as pl

class EncoderDecoder(pl.LightningModule):

    def __init__(self, encoder, decoder_embedding, decoder, decoder_outer) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder_embedding = decoder_embedding
        self.decoder = decoder
        self.decoder_outer = decoder_outer
 
    def forward(self, x, max_length, target_tensor=None):
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        encoder_hidden = self.encoder(x)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input


        x = self.encoder(x)
        
        return x

    def forward_decoder_step(self, x, hidden):
        x = self.decoder_embedding(x)
        x, hidden = self.decoder(x, hidden)
        x = self.decoder_outer(x)
        return x, hidden
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, y)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss



class SelectItem(pl.LightningModule):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
