# Importing the dependencies
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

st.set_page_config(
    page_title="Waste Segregation App",
    page_icon="♻️",
    layout="centered"
)

# Define the URL for your model file hosted on GitHub Releases
# IMPORTANT: Replace this URL with the actual link to your ResNet-50 model file
MODEL_URL = 'https://github.com/ShreyaChhabra-Innovates/Waste-Segregation-Using-CNN/releases/download/v1.0.0/resnet50_waste_model.pth'
MODEL_PATH = 'resnet50_waste_model.pth'

# This list must match the order of class indices from your training script
all_subcategories = ['food_waste', 'leaf_waste', 'paper_waste', 'wood_waste', 
                     'ewaste', 'metal_cans', 'plastic_bags', 'plastic_bottles']

# Mapping from subcategory to its main category
category_mapping = {
    'food_waste': 'biodegradable',
    'leaf_waste': 'biodegradable',
    'paper_waste': 'biodegradable',
    'wood_waste': 'biodegradable',
    'ewaste': 'non_biodegradable',
    'metal_cans': 'non_biodegradable',
    'plastic_bags': 'non_biodegradable',
    'plastic_bottles': 'non_biodegradable',
}

# --- Function to download the model ---
@st.cache_data
def download_model(url, path):
    if not os.path.exists(path):
        with st.spinner("Downloading model... this may take a moment!"):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for bad responses
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download model: {e}")
                return None
    return path

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load the model ---
@st.cache_resource
def load_model():
    model_file_path = download_model(MODEL_URL, MODEL_PATH)
    if not model_file_path:
        return None

    # Load an uninitialized ResNet-50 model
    model = models.resnet50(weights=None)

    # Replace the final fully connected layer to match the number of subcategories (8)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(all_subcategories))

    try:
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except RuntimeError as e:
        st.error(f"Error loading the model state dictionary: {e}")
        st.info("The model architecture in app.py might not match the saved model file. Check the final layer size.")
        return None

# --- Prediction function ---
def predict_image(image, model):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    predicted_subcategory = all_subcategories[predicted_class_index]
    predicted_main_category = category_mapping.get(predicted_subcategory, "Unknown")

    confidence = probabilities[predicted_class_index].item() * 100

    return predicted_main_category, predicted_subcategory, confidence

def main():
    st.title('Waste Segregation Model ♻️')
    st.markdown("""
    This application is trained using CNN transfer learning model (ResNet-50), and Cross Entropy to segregate Waste images into Biodegradable and Non-Biodegradable and their subcategories.
    Project Aim: Safe Desposal and Waste Treatment.
                
    Explore by uploading Waste images and see how it works!
                
    Github Repository: https://github.com/ShreyaChhabra-Innovates/Waste-Segregation-Using-CNN 
    """)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        model = load_model()

        if model is not None:
            predicted_main_category, predicted_subcategory, confidence = predict_image(image, model)

            st.image(image.resize((300, 300)), caption='Successfully Uploaded Image', use_container_width=True)

            st.markdown(f"**Prediction:** This is **{predicted_main_category}** waste.")
            st.markdown(f"**Subcategory:** The specific type is **{predicted_subcategory}**.")
            st.markdown(f"**Accuracy:****{confidence:.2f}%** ")

if __name__ == "__main__":
    main()
