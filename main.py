import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CIFARClassifier
from train import train_model
from evaluate import (
    evaluate_model,
    inspect_predictions,
    confusion_matrix_counts,
    most_common_confusions
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
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
    epochs=10
)

evaluate_model(
    model,
    test_loader,
    device
)

inspect_predictions(
    model,
    test_loader,
    device,
    train_dataset.classes,
    num_images=10
)

matrix = confusion_matrix_counts(
    model,
    test_loader,
    device,
    train_dataset.classes
)

most_common_confusions(
    matrix,
    train_dataset.classes,
    top_n=10
)