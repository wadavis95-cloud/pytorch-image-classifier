import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CIFARClassifier
from train import train_model
from evaluate import evaluate_model


transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFARClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs=3
)

evaluate_model(
    model,
    test_loader,
    device
)