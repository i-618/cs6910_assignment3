import torch
import pytorch_lightning as pl



class LanguageModel(pl.LightningModule):
    def __init__(self, model, loss_function, accuracy_function, optimizer_function, optimizer_params):
        super().__init__()
        self.train_accuracy = accuracy_function
        self.val_accuracy = accuracy_function
        self.loss_function = loss_function
        self.model = model
        self.optimizer = optimizer_function(self.model.parameters(), **optimizer_params)
        
    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
      # training_step defines the train loop.
      x, y = batch
      y_hat = self.forward(x)
      loss = self.loss_function(y_hat, y)
      # log step metric
      self.train_accuracy(torch.argmax(y_hat, dim=1), y)
      self.log("train_loss", loss, prog_bar=True)
      self.log("train_accuracy", self.train_accuracy, prog_bar=True)
      return loss
    
    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self.forward(x)
      loss = self.loss_function(y_hat, y)
      # log step metric
      self.val_accuracy(torch.argmax(y_hat, dim=1), y)
      self.log("val_accuracy", self.val_accuracy, on_epoch=True)
      self.log("val_loss", loss, on_epoch=True)
      return loss
    
    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute())
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return self.optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.val_accuracy.update(torch.argmax(y_hat, dim=1), y)
        self.log("test_accuracy", self.val_accuracy, on_step=False,on_epoch=True)
        self.log("test_loss", loss, on_step=False,on_epoch=True)
        return loss
    
