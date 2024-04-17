import math
import torchmetrics
from languageModel import LanguageModel
import torch
import torchvision as tv
import pytorch_lightning as pl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = tv.datasets.ImageFolder(
     root='inaturalist_12K/train', transform=tv.transforms.Compose([
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # tv.transforms.Lambda(lambda x: x.to(device)),
    tv.transforms.Resize((300, 300)),
]),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])




conv_actv_maxout = [torch.nn.Conv2d(15, 15, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2)]*3
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 6, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),

    torch.nn.Conv2d(6, 16, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),

    torch.nn.Conv2d(16, 32, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),

    torch.nn.Flatten(1, -1),

)
input_dim = train_data[0][0].shape
num_features_before_fcnn = math.prod(list(model(torch.rand(1, *input_dim)).shape))
print('num_features_before_fcnn', num_features_before_fcnn)
model.extend([
    torch.nn.Linear(num_features_before_fcnn, 10),
    ])

optimizer_function = torch.optim.Adam
optimizer_params = {}
accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()

model = LanguageModel(model=model, loss_function=loss_function, accuracy_function=accuracy_function, 
                        optimizer_function=optimizer_function, optimizer_params=optimizer_params)


# model.to(device)
print(model)
trainer  = pl.Trainer(log_every_n_steps=5, max_epochs=100)
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=int(len(train_data)/3))
val_dataloaders = torch.utils.data.DataLoader(val_data)

trainer.fit( model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)





