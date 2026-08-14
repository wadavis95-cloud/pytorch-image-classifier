import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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

for i in range(5):
    image, label = train_dataset[i]

    image = image.permute(1, 2, 0)

    plt.figure()
    plt.imshow(image)
    plt.title(train_dataset.classes[label])
    plt.axis("off")

    plt.show()

print("\nClass labels")

for i, class_name in enumerate(train_dataset.classes):
    print(i, "->", class_name)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("First 10 labels:", labels[:10])