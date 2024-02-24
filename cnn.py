import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchsummary import summary
from my_utils import eta

import wandb

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = 0.2
val_samples = int(val_size * len(train_dataset))
train_samples = len(train_dataset) - val_samples

train_dataset, val_dataset = random_split(train_dataset, [train_samples, val_samples])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

batch_size = 64
learning_rate = 0.01
num_epochs = 15
momentum = 0.8

kernel_size = 2  # Decrease the kernel size to 2

dropout_rate = 0.4
l1_size = 256
l2_size = 128

conv_1_size = 32
conv_2_size = 32
conv_3_size = 128
conv_4_size = 0

train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# plt.imshow(train_dataset.dataset.data[0], cmap='gray')
# plt.axis("off")
# plt.show()

wandb.init(
    project="mnist-cnn",
    config={
        "data_normalization": "True",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "momentum": momentum,
        "kernel_size": kernel_size,
        "optimizer": "SGD with momentum",
        "dropout_rate": dropout_rate,
        "l1_size": l1_size,
        "l2_size": l2_size,
        "batch_norm": 1,
        "conv_1_size": conv_1_size,
        "conv_2_size": conv_2_size,
        "conv_3_size": conv_3_size,
        "conv_4_size": conv_4_size,
    })

class CNN(nn.Module):
    def __init__(self, l1_size=120, l2_size=84, kernel_size=3, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_1_size, kernel_size) # 1 input, conv_1_size output, kernel_size kernel size
        self.conv2 = nn.Conv2d(conv_1_size, conv_2_size, kernel_size)
        self.conv3 = nn.Conv2d(conv_2_size, conv_3_size, kernel_size)
        # self.conv4 = nn.Conv2d(conv_3_size, conv_4_size, kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.maxPool = nn.MaxPool2d(2, 2) # 2 kernel size, 2 stride, no padding
        self.fc1 = nn.Linear(512, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, 10)
        self.batch_norm1 = nn.BatchNorm2d(conv_1_size)
        self.batch_norm2 = nn.BatchNorm2d(conv_2_size)

    def forward(self, x):
        x = F.relu(self.maxPool(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.maxPool(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.maxPool(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) 

 
model = CNN(l1_size, l2_size, kernel_size, dropout_rate).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(type(train.dataset.dataset.data[0]), train.dataset.dataset.data[0].shape)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_correct = 0
    total_train_samples = 0

    for i, (X, y) in enumerate(train):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_train_samples += y.size(0)
        
        eta(epoch, i, num_epochs, train)
        

    avg_train_loss = total_train_loss / len(train)
    train_accuracy = train_correct / total_train_samples
    
    wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy})
    print(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    model.eval()
    total_val_loss = 0
    val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(val):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            total_val_loss += loss.item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_val_samples += y.size(0)
            wandb.log({"val_loss": total_val_loss / len(val), "val_accuracy": val_correct / total_val_samples})
        print(f"Validation Loss: {total_val_loss / len(val):.4f}, Validation Accuracy: {val_correct / total_val_samples:.4f}")

# wandb.finish()