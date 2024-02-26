from models import create_cnn_model
from trainloop import train_model, init_wandb
from hyperparameters import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MOMENTUM, KERNEL_SIZE, DROPOUT_RATE, L1_SIZE, L2_SIZE, CONV_1_SIZE, CONV_2_SIZE, CONV_3_SIZE, CONV_4_SIZE, DATA_NORMALIZATION, OPTIMIZER
import torch
from datasets import train_grayscale, val_grayscale
import sys

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

for sys_arg in sys.argv:
    if sys_arg in config.keys():
        key, value = sys_arg.split("=")
        config[key[2:]] = value
    else:
        print(f"Invalid argument: {sys_arg}. Ignoring.")

init_wandb("mnist-cnn", config)

loss_fn = torch.nn.CrossEntropyLoss()

print("Training CNN model with the following configuration:")
for key, value in config.items():
    print(f"{key}: {value}")

model = create_cnn_model(l1_size=config["l1_size"],
                            l2_size=config["l2_size"],
                            kernel_size=config["kernel_size"],
                            dropout_rate=config["dropout_rate"],
                            conv_1_size=config["conv_1_size"],
                            conv_2_size=config["conv_2_size"],
                            conv_3_size=config["conv_3_size"],
                            conv_4_size=config["conv_4_size"]
                         )

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

trained_model = train_model(model, loss_fn, optimizer, train_grayscale, val_grayscale)
