import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import wandb
from my_utils import eta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# train_dataset.data = train_dataset.data.unsqueeze(1).repeat(1, 3, 1, 1)
# test_dataset.data = test_dataset.data.unsqueeze(1).repeat(1, 3, 1, 1)

val_size = 0.2
val_samples = int(val_size * len(train_dataset))
train_samples = len(train_dataset) - val_samples

train_dataset, val_dataset = random_split(train_dataset, [train_samples, val_samples])

batch_size = 128
learning_rate = 0.001
num_epochs = 10
momentum = 0.9

train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 10).to(device)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# imshow(torchvision.utils.make_grid(train_dataset.dataset.data[0:4]))
    
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

wandb.init(
    project="mnist-resnet",
    config={
        "data_normalization": "True",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "momentum": momentum,
        "optimizer": "SGD with momentum",
    }
)

wandb.watch(model, log="all")

# print(type(train.dataset.dataset.data[0]), train.dataset.dataset.data[0].shape)

def train_model(model, loss_fn, optimizer, num_epochs):
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

    return model.to(device)
    #  .finish()

model = train_model(model, loss_fn, optimizer, num_epochs)

def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

train_accuracy = test_model(model, train)
val_accuracy = test_model(model, val)

wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})

print(f'Train accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}')