import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import CIFARClassifier


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = CIFARClassifier().to(device)

model.load_state_dict(
    torch.load(
        "models/cifar_classifier.pth",
        map_location=device
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image"
    )

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    _, predicted = torch.max(outputs, 1)

    prediction = classes[predicted.item()]

    st.subheader(
        f"Prediction: {prediction}"
    )