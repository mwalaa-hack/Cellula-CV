
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cpu")
NUM_CLASSES = 7

CLASS_NAMES = [
    "CaS",
    "CoS",
    "Gum",
    "MC",
    "OC",
    "OLP",
    "OT"
]

#load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load("resnet_transferL2.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# UI
st.title("Teeth Disease Classification")

gui_image = Image.open("picture.webp")
st.image(gui_image, use_container_width=True)


uploaded_file = st.file_uploader("Upload an image of a tooth/disease", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")
