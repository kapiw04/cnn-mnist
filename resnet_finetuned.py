from models import resnet50Model
from trainloop import train_model, init_wandb
from hyperparameters import *
import torch
from datasets import train_RGB, val_RGB

config = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "momentum": MOMENTUM,
}

init_wandb("resnet50", config)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet50Model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

trained_model = train_model(resnet50Model, loss_fn, optimizer, train_RGB, val_RGB)
