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

# --- STEP 1: INITIALIZE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="VAE Latent Explorer", layout="wide")

CELEBA_ATTRIBUTES = {
    "Eyeglasses": 15, "Goatee": 16, "Mustache": 22, "Bald": 4, "Young": 39,
    "Smiling": 31, "Male": 20, "Straight_Hair": 32, "Wavy_Hair": 33, "Blond_Hair": 9
}

# --- STEP 2: CACHED RESOURCE INITIALIZATION ---
@st.cache_resource
def get_dataloader():
    """Initializes and caches the underlying CelebA pipeline data stream."""
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
    
    # 64 batch size speeds up the initial subset slicing loop
    return DataLoader(celeb_test_dataset, shuffle=True, batch_size=64, num_workers=0, pin_memory=False)


@st.cache_resource
def get_model_and_vectors(_testloader):
    """Loads weights and pre-computes attribute direction vectors using a data subset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE()
    vae.to(device)
    vae.eval()

    checkpoint = torch.load("weights/vae_35.pth", map_location=device)
    vae.load_state_dict(checkpoint['vae'])

    latents = []
    features = []

    with torch.no_grad():
        for batch in _testloader:
            images = batch[0].to(device)
            attributes = batch[1]

            mean_images = vae.mean(vae.encoder(images))

            latents.append(mean_images.cpu().numpy())
            features.append(attributes.numpy())
            
            # Sub-sample optimization limit for fluid server startup speeds
            if len(latents) * 64 >= 4000:
                break

        latents_np = np.concatenate(latents, axis=0)
        features_np = np.concatenate(features, axis=0)

    latent_vectors = {} 

    for key, val in CELEBA_ATTRIBUTES.items():
        positive_mask = features_np[:, val] == 1
        negative_mask = features_np[:, val] == 0

        # Protect against empty masks if utilizing highly restrictive filters
        if np.sum(positive_mask) > 0 and np.sum(negative_mask) > 0:
            vector = np.mean(latents_np[positive_mask], axis=0) - np.mean(latents_np[negative_mask], axis=0)
            vector_norm = vector / (np.linalg.norm(vector) + 1e-8)
        else:
            vector_norm = np.zeros(latents_np.shape[1])
            
        latent_vectors[key] = torch.tensor(vector_norm, dtype=torch.float32).to(device)

    return vae, latent_vectors, device

# Resolve cached pipeline layers
testloader = get_dataloader()
vae, latent_vectors, device = get_model_and_vectors(testloader)


# --- STEP 3: LATENT STATE WORKERS ---
def get_random_base_latent():
    """Extracts a singular baseline feature profile mapping target from pipeline."""
    batch_images, _ = next(iter(testloader))
    random_idx = random.randint(0, batch_images.shape[0] - 1)
    random_single_img = batch_images[random_idx].unsqueeze(0).to(device)
    
    vae.eval()
    with torch.no_grad():
        z_random_base = vae.mean(vae.encoder(random_single_img))
    return z_random_base


# Enforce safe baseline profile structure initialization across app runs
if "z_base" not in st.session_state:
    st.session_state.z_base = get_random_base_latent()


def load_new_face_callback():
    """Mutates baseline state configuration targets and forces slider state reset."""
    st.session_state.z_base = get_random_base_latent()
    # Explicitly snap UI component widgets back to zero baseline point
    for feature_name in CELEBA_ATTRIBUTES.keys():
        st.session_state[f"slider_{feature_name}"] = 0.0


# --- STEP 4: UI AND CSS VIEW RENDERING ---
st.markdown("""
    <style>
    .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 8px;
        max-height: 320px;
        object-fit: contain;
        /* Preserves pixel edges to stop blurriness */
        image-rendering: pixelated; 
    }
    h3 { text-align: center; font-family: sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### VAE Facial Modification Dashboard")

col_sliders, col_display = st.columns([1, 1])

slider_values = {}
with col_sliders:
    for feature_name in CELEBA_ATTRIBUTES.keys():
        # Inject standard value keys into state tracking pools if missing
        if f"slider_{feature_name}" not in st.session_state:
            st.session_state[f"slider_{feature_name}"] = 0.0
            
        val = st.slider(
            label=feature_name,
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            key=f"slider_{feature_name}"  
        )
        slider_values[feature_name] = val

with col_display:
    with torch.no_grad():
        # Vectorized latent canvas generation step
        z_modified = st.session_state.z_base.clone()
        
        # Shift latent matrix coordinates cleanly based on active slider positions
        for feature_name, value in slider_values.items():
            z_modified += float(value) * latent_vectors[feature_name]
            
        generated_output = vae.decoder(vae.decoder_input(z_modified))
        
        # Rescale image pixels safely to clean PIL storage range [0.0 - 1.0]
        img_tensor = (generated_output.squeeze(0) + 1.0) / 2.0
        img_tensor = img_tensor.clamp(0.0, 1.0)
        output_pil_image = T.ToPILImage()(img_tensor.cpu())

    st.image(output_pil_image, caption="Generated Output Face", use_container_width=True)
    
    st.button(
        label="Load Random Face 👤", 
        type="primary", 
        on_click=load_new_face_callback,
        use_container_width=True
    )