import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

print("Number of training images:", len(train_dataset))
print("Classes:", train_dataset.classes)

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label number:", label)
print("Label name:", train_dataset.classes[label])