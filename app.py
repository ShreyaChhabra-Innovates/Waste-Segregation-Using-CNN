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

# IMPORTANT: Replace this URL with the actual link to your trained ResNet-50 model file
MODEL_URL = 'https://github.com/your-username/your-repo/releases/download/v2.0.0/resnet50_waste_new_model.pth'
MODEL_PATH = 'resnet50_waste_new_model.pth'

# This list must exactly match the order of class names from your training script
all_subcategories = ['ewaste', 'food_waste', 'leaf_waste', 'metal_cans', 
                     'paper_waste', 'plastic_bags', 'plastic_bottles', 'wood_waste']

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

# --- Function to download the model from a URL ---
@st.cache_data
def download_model(url, path):
    """Downloads the model file from a URL if it doesn't already exist."""
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

# --- Load the model based on the ResNet-50 architecture ---
@st.cache_resource
def load_model(model_path):
    """
    Loads the trained ResNet-50 model with the correct architecture.
    """
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check the download URL and local path.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load an uninitialized ResNet-50 model
    model = models.resnet50(weights=None)

    # Replace the final fully connected layer to match the number of subcategories (8)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(all_subcategories))

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success("Model loaded successfully!")
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading the model state dictionary: {e}")
        st.info("The model architecture might not match the saved model file. Check the final layer size.")
        return None

# --- Prediction function for multi-class classification ---
def predict_image(image, model):
    """
    Makes a prediction on a single image and returns the top 3 predictions.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    device = next(model.parameters()).device
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get the top 3 predictions
    top_prob, top_indices = torch.topk(probabilities, 3)

    results = []
    for i in range(len(top_indices)):
        subcategory = all_subcategories[top_indices[i].item()]
        main_category = category_mapping.get(subcategory, "Unknown")
        confidence = top_prob[i].item() * 100
        results.append({
            'main_category': main_category,
            'subcategory': subcategory,
            'confidence': confidence
        })
    return results

def main():
    """Main function to run the Streamlit app"""
    st.title('Waste Segregation Model ♻️')
    st.markdown("""
    This application uses a trained **ResNet-50** model to classify waste images into specific subcategories
    and their main type (Biodegradable or Non-Biodegradable).
    """)

    # Download and load the model once
    model_file = download_model(MODEL_URL, MODEL_PATH)
    if model_file:
        model = load_model(model_file)
    else:
        model = None
    
    if model is None:
        st.warning("Model could not be loaded. Please check the URL and your file.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Make a prediction
        top_predictions = predict_image(image, model)

        # Display results
        st.image(image, caption='Successfully Uploaded Image', use_column_width=True)
        
        st.subheader("Top Predictions:")
        for i, pred in enumerate(top_predictions):
            st.markdown(f"**{i+1}. {pred['main_category'].capitalize()} ({pred['subcategory']})** with **{pred['confidence']:.2f}%** confidence.")

if __name__ == "__main__":
    main()

