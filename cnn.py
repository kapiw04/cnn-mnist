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


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

val_size = 0.2
val_samples = int(val_size * len(train_dataset))

train_samples = len(train_dataset) - val_samples

train_dataset, val_dataset = random_split(train_dataset, [train_samples, val_samples])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

batch_size = 32
learning_rate = 0.01
num_epochs = 10
momentum = 0.9

kernel_size = 3

train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# plt.imshow(train_dataset.dataset.data[0], cmap='gray')
# plt.axis("off")
# plt.show()

wandb.init(
    project="mnist-cnn",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "momentum": momentum,
        "kernel_size": kernel_size
    })

class CNN(nn.Module):
    def __init__(self, l1_size=120, l2_size=84) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size) # 1 input, 6 output, kernel_size kernel size
        self.maxPool = nn.MaxPool2d(2, 2) # 2 kernel size, 2 stride, no padding
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.fc1 = nn.Linear(16 * 5 * 5, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, 10)

    def forward(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

 
model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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

wandb.finish()