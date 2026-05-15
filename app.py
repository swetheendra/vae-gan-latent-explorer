import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as T
import random
from model_utils import VAE, AlbumentationWrapper

st.set_page_config(page_title="VAE Latent Explorer", layout="wide")

CELEBA_ATTRIBUTES = {
    "Eyeglasses": 15, "Goatee": 16, "Mustache": 22, "Bald": 4, "Young": 39,
    "Smiling": 31, "Male": 20, "Straight_Hair": 32, "Wavy_Hair": 33, "Blond_Hair": 9
}

# --- STEP 2: CACHED DATA LOADING (OPTIMIZED) ---
@st.cache_resource
def get_model_and_vectors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE()
    vae.to(device)

    checkpoint = torch.load("weights/vae_35.pth", map_location=device)
    vae.load_state_dict(checkpoint['vae'])

    transform = A.Compose([
        A.Resize(height=64, width=64),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    celeb_test_dataset = datasets.CelebA(
        root='data/celeba',
        split='test',
        target_type='attr',
        download=True,
        transform=AlbumentationWrapper(transform)
    )

    # Use a larger batch size for faster startup feature gathering
    testloader = DataLoader(celeb_test_dataset, shuffle=True, batch_size=64, num_workers=0, pin_memory=False)

    latents = []
    features = []

    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            attributes = batch[1]

            mean_images = vae.mean(vae.encoder(images))

            latents.append(mean_images.cpu().numpy())
            features.append(attributes.numpy())
            
            # CRUCIAL OPTIMIZATION: Stop after 4000 images so startup is instant
            if len(latents) * 64 >= 4000:
                break

        latents_np = np.concatenate(latents, axis=0)
        features_np = np.concatenate(features, axis=0)

    latent_vectors = {} 

    for key, val in CELEBA_ATTRIBUTES.items():
        positive_mask = features_np[:, val] == 1
        negative_mask = features_np[:, val] == 0

        vector = np.mean(latents_np[positive_mask], axis=0) - np.mean(latents_np[negative_mask], axis=0)
        latent_vectors[key] = torch.tensor(vector / np.linalg.norm(vector), dtype=torch.float32).to(device)

    return vae, latent_vectors, device, testloader

vae, latent_vectors, device, testloader = get_model_and_vectors()


def get_random_base_latent():
    # Fetch a random batch from the loader
    batch_images, _ = next(iter(testloader))
    random_idx = random.randint(0, batch_images.shape[0] - 1)
    random_single_img = batch_images[random_idx].unsqueeze(0).to(device)
    
    vae.eval()
    with torch.no_grad():
        z_random_base = vae.mean(vae.encoder(random_single_img))
    return z_random_base


# Initialize your baseline face tracking session coordinate target safely
if "z_base" not in st.session_state:
    st.session_state.z_base = get_random_base_latent()

# Reset button action logic mapping
def load_new_face_callback():
    st.session_state.z_base = get_random_base_latent()
    # Resetting keys via a separate state clear to prevent render-clashes
    for feature_name in CELEBA_ATTRIBUTES.keys():
        if f"slider_{feature_name}" in st.session_state:
            del st.session_state[f"slider_{feature_name}"]

# --- UI STYLING INJECTORS ---
st.markdown("""
    <style>
    .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 6px;
        max-height: 280px;
        object-fit: contain;
    }
    h3 { text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### VAE Facial Modification Dashboard")

# 3. Double-Column Layout Architecture Mapping
col_sliders, col_display = st.columns([1, 1])

slider_values = []
with col_sliders:
    for feature_name in CELEBA_ATTRIBUTES.keys():
        val = st.slider(
            label=feature_name,
            min_value=-3.0,
            max_value=3.0,
            value=0.0, # Default starting point
            step=0.1,
            key=f"slider_{feature_name}"  
        )
        slider_values.append(val)

with col_display:
    with torch.no_grad():
        z_modified_1d = st.session_state.z_base.squeeze(0).clone()
        
        for value, feature_name in zip(slider_values, CELEBA_ATTRIBUTES.keys()):
            z_modified_1d += float(value) * latent_vectors[feature_name]
            
        z_final = z_modified_1d.unsqueeze(0)
        generated_output = vae.decoder(vae.decoder_input(z_final))
        
        img_tensor = (generated_output.squeeze(0) + 1.0) / 2.0
        img_tensor = img_tensor.clamp(0.0, 1.0)
        output_pil_image = T.ToPILImage()(img_tensor.cpu())

    st.image(output_pil_image, caption="Generated Output Face", width=280)
    
    st.button(
        label="Load Random Face 👤", 
        type="primary", 
        on_click=load_new_face_callback,
        use_container_width=True
    )
