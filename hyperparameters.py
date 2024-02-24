import torch

BATCH_SIZE = 32
LEARNING_RATE = 0.008
NUM_EPOCHS = 15
MOMENTUM = 0.8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.5
L1_SIZE = 256
L2_SIZE = 128
CONV_1_SIZE = 32
CONV_2_SIZE = 32
CONV_3_SIZE = 0
CONV_4_SIZE = 0
DATA_NORMALIZATION = "True"
OPTIMIZER = "SGD with momentum"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")