import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import gradio as gr
import torchvision.transforms as T


class VAE(nn.Module):
  def __init__(self, input_dim=3,latent_dim=128):
    super().__init__()

    self.encoder = nn.Sequential(
          nn.Conv2d(input_dim, 64, padding=1, kernel_size=3, stride=2),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 128, padding=1, kernel_size=3, stride=2),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 256, padding=1, kernel_size=3, stride=2),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 512, padding=1, kernel_size=3, stride=2),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(512*4*4, 512)
        )

    self.mean = nn.Linear(512, latent_dim)
    self.logvar = nn.Linear(512, latent_dim)

    self.decoder_input = nn.Linear(latent_dim, 512*4*4)

    self.decoder = nn.Sequential(
        nn.Unflatten(1, (512, 4, 4)),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )

  def reparametrize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu+std * eps

  def forward(self, x):
    x = self.encoder(x)
    mu, logvar = self.mean(x), self.logvar(x)
    z = self.reparametrize(mu, logvar)

    decoder_input = self.decoder_input(z)
    output = self.decoder(decoder_input)

    return output, mu, logvar

class Discriminator(nn.Module):

  def __init__(self, input_dim=3):
    super().__init__()
    self.main = nn.Sequential(
        nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256,1,kernel_size=3, stride=2, padding=1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.main(x).view(-1,1)

class PerceptualLoss(nn.Module):
  def __init__(self):
    super().__init__()
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    vgg = vgg.to(device)
    self.slices = nn.Sequential(*vgg[:16]).eval()
    for param in self.slices.parameters():
            param.requires_grad = False
  def forward(self, x,y):
    return F.mse_loss(self.slices(x), self.slices(y))


class AlbumentationWrapper:

  def __init__(self, transform):
    self.transform = transform

  def __call__(self, img):
    return self.transform(image=np.array(img))['image']