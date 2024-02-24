from models import cnnModel
from trainloop import train_model, init_wandb
from hyperparameters import *
import torch
from datasets import train_grayscale, val_grayscale

config = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "momentum": MOMENTUM,
    "kernel_size": KERNEL_SIZE,
    "dropout_rate": DROPOUT_RATE,
    "l1_size": L1_SIZE,
    "l2_size": L2_SIZE,
    "conv_1_size": CONV_1_SIZE,
    "conv_2_size": CONV_2_SIZE,
    "conv_3_size": CONV_3_SIZE,
    "conv_4_size": CONV_4_SIZE,
    "data_normalization": DATA_NORMALIZATION,
    "optimizer": OPTIMIZER,
}

init_wandb("resnet50", config)

loss_fn = torch.nn.CrossEntropyLoss()

model = cnnModel

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

trained_model = train_model(model, loss_fn, optimizer, train_grayscale, val_grayscale)
