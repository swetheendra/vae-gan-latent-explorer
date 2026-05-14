import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as T
from model_utils import VAE, AlbumentationWrapper


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

testloader = DataLoader(celeb_test_dataset, shuffle=True, batch_size=4, num_workers=3, pin_memory=True)

CELEBA_ATTRIBUTES = {
    "Eyeglasses": 15, "Goatee": 16, "Mustache": 22, "Bald": 4, "Young": 39,
    "Smiling": 31, "Male": 20, "Straight_Hair": 32, "Wavy_Hair": 33, "Blond_Hair": 9
}


latents = []
features = []

with torch.no_grad():
    for batch in testloader:
        images = batch[0]
        attributes = batch[1]

        images = images.to(device)

        mean_images = vae.mean(vae.encoder(images))

        latents.append(mean_images.cpu().numpy())
        features.append(attributes.cpu().numpy())

    latents_np = np.concatenate(latents, axis=0)
    features_np = np.concatenate(features, axis=0)

latent_vectors = {} 

for key,val in CELEBA_ATTRIBUTES.items():
    positive_mask = features_np[:,val] == 1
    negative_mask = features_np[:,val] == 0

    vector = np.mean(latents_np[positive_mask], axis=0) - np.mean(latents_np[negative_mask], axis=0)
    latent_vectors[key] = torch.tensor(vector / np.linalg.norm(vector), dtype=torch.float32).to(device)


def get_random_base_latent():
    """
    Grabs a completely random single image from the validation dataset loader
    and calculates its base latent vector profile.
    """
    import random
    
    # 1. Convert loader to an iterable list or pick a random batch index
    # Iterating over the loader lets us pull a dynamic fresh batch
    batch_images, _ = next(iter(testloader))
    
    # 2. Pick a random image index position inside that batch
    random_idx = random.randint(0, batch_images.shape[0] - 1)
    random_single_img = batch_images[random_idx].unsqueeze(0).to(device)
    
    # 3. Calculate and return its base latent vector profile
    vae.eval()
    with torch.no_grad():
        z_random_base = vae.mean(vae.encoder(random_single_img))
    
    return z_random_base


# Initialize data pipeline and cache your 10 direction vectors on startup
if 'vectors_loaded' not in st.session_state:
    # (Optional) Initialize your validation data loader loop step here
    # initialize_latent_vectors(val_loader)
    st.session_state.vectors_loaded = True

# Initialize your baseline face tracking session coordinate target
if "z_base" not in st.session_state:
    st.session_state.z_base = get_random_base_latent(val_loader)

# 1. UI Page & CSS Alignment Tuning
st.set_page_config(page_title="VAE Latent Explorer", layout="wide")

# Center and frame the generated output image crisply using Streamlit CSS injectors
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
""", unsafe_allow_value=True)

st.markdown("### VAE Facial Modification Dashboard")

# 2. Session State Initialization (Locks identity between slider movements)
if "z_base" not in st.session_state:
    # Initialize your model's baseline coordinate on first application boot
    st.session_state.z_base = get_random_base_latent()

# Reset button action logic mapping
def load_new_face_callback():
    # Force state mutation directly via session caching variables
    st.session_state.z_base = get_random_base_latent()
    
    # We clear the specific slider state keys so they physically jump back to 0.0
    for feature_name in CELEBA_ATTRIBUTES.keys():
        st.session_state[f"slider_{feature_name}"] = 0.0

# 3. Double-Column Layout Architecture Mapping
col_sliders, col_display = st.columns([1, 1])

# Left Column: Input Panel (Generates 10 sliders dynamically)
slider_values = []
with col_sliders:
    for feature_name in CELEBA_ATTRIBUTES.keys():
        # Using explicit keys binds the components to our callback reset clear routine
        val = st.slider(
            label=feature_name,
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            key=f"slider_{feature_name}"  # Important for session clearing
        )
        slider_values.append(val)

# Right Column: Visualizer & Control Action Blocks
with col_display:
    # Core Mathematical Latent Modification Function Pipeline Block
    with torch.no_grad():
        # 1. Strip batch dimensions using your proven 1D isolated vector arithmetic setup
        z_modified_1d = st.session_state.z_base.squeeze(0).clone()
        
        # 2. Shift latent matrix coordinates cleanly based on active slider data
        for value, feature_name in zip(slider_values, CELEBA_ATTRIBUTES.keys()):
            z_modified_1d += float(value) * latent_vectors[feature_name]
            
        # 3. Reshape and route forward step calls out to VAE decoder layers
        z_final = z_modified_1d.unsqueeze(0)
        generated_output = vae.decoder(vae.decoder_input(z_final))
        
        # 4. Normalize pixel arrays and project down to a clean PIL container
        img_tensor = (generated_output.squeeze(0) + 1.0) / 2.0
        img_tensor = img_tensor.clamp(0.0, 1.0)
        output_pil_image = T.ToPILImage()(img_tensor.cpu())

    # Display the processed high-fidelity image component frame
    st.image(output_pil_image, caption="Generated Output Face", use_column_width=False, width=280)
    
    # Render Action Reset trigger button right under the image viewer
    st.button(
        label="Load Random Face 👤", 
        type="primary", 
        on_click=load_new_face_callback,
        use_container_width=True
    )
