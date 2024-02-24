from torchvision import transforms
import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from hyperparameters import BATCH_SIZE

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_grayscale = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset_RGB = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset_RGB = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset_grayscale = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_grayscale)
test_dataset_grayscale = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_grayscale)

# train_dataset.data = train_dataset.data.unsqueeze(1).repeat(1, 3, 1, 1)
# test_dataset.data = test_dataset.data.unsqueeze(1).repeat(1, 3, 1, 1)

val_size = 0.2
val_samples = int(val_size * len(train_dataset_RGB))
train_samples = len(train_dataset_RGB) - val_samples

train_dataset_RGB, val_dataset_RGB = random_split(train_dataset_RGB, [train_samples, val_samples])
train_dataset_grayscale, val_dataset_grayscale = random_split(train_dataset_grayscale, [train_samples, val_samples])

train_RGB = DataLoader(train_dataset_RGB, batch_size=BATCH_SIZE, shuffle=True)
val_RGB = DataLoader(val_dataset_RGB, batch_size=BATCH_SIZE, shuffle=True)
test_RGB = DataLoader(test_dataset_RGB, batch_size=BATCH_SIZE, shuffle=True)

train_grayscale = DataLoader(train_dataset_grayscale, batch_size=BATCH_SIZE, shuffle=True)
val_grayscale = DataLoader(val_dataset_grayscale, batch_size=BATCH_SIZE, shuffle=True)
test_grayscale = DataLoader(test_dataset_grayscale, batch_size=BATCH_SIZE, shuffle=True)
