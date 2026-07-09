# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:51:34 2024

@author: nadja 
"""
#%%

# %%
import torch
import cv2 as cv
from numbers import Number
import os
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import skimage.metrics as skm
from skimage.data import shepp_logan_phantom
import logging
import numpy as np
from tomosipo.torch_support import (
    to_autograd,
)
from models import *
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import LION.CTtools.ct_geometry as ctgeo
from skimage.transform import rescale, resize
import skimage
import argparse
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import gc
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch.nn.functional as F
from scipy.interpolate import griddata
import itertools
#%%

'>>-------------------------------------------------------------------------<<'
' Define helper functions'
'>>-------------------------------------------------------------------------<<'

def add_gaussian_noise(img, sigma):
    #img = np.array(img)
    noise = torch.normal(0, sigma, img.shape, device = img.device) / 100
    img = img + torch.max(img) * noise
    return img

##### poisson noise
def apply_poisson_noise(img, photon_count, seed=41):
    rng = np.random.default_rng(seed)  # reproducible RNG
    opt = dict(dtype=np.float32)

    # Convert to transmission domain
    img = np.exp(-img.detach().cpu().numpy(), **opt)

    # Add Poisson noise with fixed seed
    noisy_img = rng.poisson(img * photon_count).astype(np.float32)
    noisy_img[noisy_img == 0] = 1  # avoid log(0)
    noisy_img /= photon_count

    # Convert back to attenuation domain
    noisy_img = -np.log(noisy_img, **opt)

    # Convert back to torch tensor if needed
    return torch.from_numpy(noisy_img)


def create_noisy_sinograms_poisson(
    images,
    angles_full,
    attenuation_factor=2.76,
    photon_count=3000,
    correlated_noise=True,
    kernel_size=5,
    sigma_blur=3.0,
    gaussian_noise_std=0.1,   # std of gaussian noise added in correlated noise case
):
    """
    gaussian_noise_std controls additive Gaussian noise strength.
    """

    # --- 1. Geometry ---
    geo = ctgeo.Geometry.parallel_default_parameters(image_shape=images.shape)
    geo.angles = np.linspace(0, np.pi, angles_full, endpoint=False)
    geo.image_pos = np.array([0.0, 0.0, 0.0])

    # --- 2. Forward operator ---
    op = ct.make_operator(geo)
    sino = op(images)                      # [B,H,A]

    maxi = torch.max(sino)
    sino = sino / maxi * attenuation_factor

    # --- 3. Noise ---
    if not correlated_noise:
        proj_noisy = apply_poisson_noise(sino, photon_count)

    else:
        B, H, A = sino.shape

        # --- white Gaussian noise with user strength ---
        sigma = gaussian_noise_std * torch.mean(sino)
        noise = torch.randn_like(sino) * sigma

        # reshape to conv2d format [B,1,A,H]
        noise_2d = noise.permute(0, 2, 1).unsqueeze(1)

        # --- fixed vertical Gaussian kernel ---
        x = torch.arange(kernel_size, device=sino.device) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * sigma_blur**2))
        kernel = kernel / kernel.sum()

        weight = torch.zeros(1, 1, kernel_size, 1, device=sino.device)
        weight[0, 0, :, 0] = kernel

        conv = torch.nn.Conv2d(
            1, 1,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=False
        ).to(sino.device)

        with torch.no_grad():
            conv.weight.copy_(weight)
        conv.requires_grad_(False)

        # --- correlate noise vertically ---
        noise_corr = conv(noise_2d)

        # back to [B,H,A]
        noise_corr = noise_corr.squeeze(1).permute(0, 2, 1)

        proj_noisy = sino + noise_corr

    # --- 4. Undo normalization ---
    proj_noisy = proj_noisy / attenuation_factor * maxi.to(proj_noisy.device)

    # --- 5. Final formatting ---
    sinogram_full = torch.moveaxis(proj_noisy, -1, -2)
    return sinogram_full.detach().unsqueeze(1)




##### create gaussian noise sinograms
def create_noisy_sinograms(images, angles_full, sigma):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape
    )  # parallel beam standard CT
    # 0.2: create operator:
    geo.angles = np.linspace(0, np.pi, angles_full, endpoint=False) 
    geo.image_pos = np.array([0.0, 0.0, 0.0])
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(images)
    sinogram_full = add_gaussian_noise(sino, sigma)
    sinogram_full = torch.moveaxis(sinogram_full, -1, -2)
    return sinogram_full.unsqueeze(1)

def sobolev_norm(f, g):
    l2 = torch.nn.functional.mse_loss(f, g)
    im_size = f.shape[-2]
    ### f and g have shape (batch size, 1, N_s, N_theta)
    derivatives_s1 = torch.gradient(f, axis=2)[0]
    derivatives_s2 = torch.gradient(g, axis=2)[0]
    l2_grad = torch.nn.functional.mse_loss(derivatives_s1, derivatives_s2)
    # print("gradient term is: " + str(l2_grad))
    # print("l2 term is: " + str(l2))
    sobolev = (l2**2 + l2_grad**2) / im_size**2
    return sobolev

def sobolev_norm_fourier(f, g, s = 1, a = 1):
    ### f and g have shape (batch size, 1, N_s, N_theta)
    Ff = torch.fft.fft(f,dim = 2)
    Fg = torch.fft.fft(g,dim = 2)
    freq = torch.fft.fftfreq(n = f.shape[-2])
    Fdiff = Ff-Fg
    Ffreq = (a+torch.abs(freq)**2)**(s/2)
    Ffreq = Ffreq.view(1, 1, -1, 1)
    loss = torch.mean(torch.abs((Ffreq.to(f.device)*Fdiff)**2))
    return loss

#%%

def fill_zero_columns(tensor, zeros_in='odd', fill_edges=False):
    """
    Fill zero columns with the mean of their neighbors.

    Args:
        tensor (torch.Tensor): Input tensor of shape (..., H, W)
        zeros_in (str): 'odd' if zeros are in odd columns (1,3,5,...),
                        'even' if zeros are in even columns (0,2,4,...)
        fill_edges (bool): If True, fills edge zero columns by copying nearest neighbor.
        
    Returns:
        torch.Tensor: Tensor with zero columns filled.
    """
    x = tensor.clone()
    
    if zeros_in == 'odd':
        cols = torch.arange(1, x.shape[-1], 2)
    elif zeros_in == 'even':
        cols = torch.arange(0, x.shape[-1], 2)
    else:
        raise ValueError("zeros_in must be 'odd' or 'even'")
    
    # columns with both neighbors available
    cols_valid = cols[(cols > 0) & (cols < x.shape[-1] - 1)]
    left = x[..., cols_valid - 1]
    right = x[..., cols_valid + 1]
    x[..., cols_valid] = (left + right) / 2

    if fill_edges:
        # first zero column if it's at index 0
        if 0 in cols:
            x[..., 0] = x[..., 1]
        # last zero column if it's at last index
        if (x.shape[-1] - 1) in cols:
            x[..., -1] = x[..., -2]

    return x

# %% function for reading in our walnut data
def get_images(path, amount_of_images="all", scale_number=1):
    all_images = []
    all_image_names = os.listdir(path)
    if amount_of_images == "all":
        for name in all_image_names:
            temp_image = cv.imread(path + "/" + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]

        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + "/" + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)

    return np.array(all_images, dtype="float16")



def create_circular_mask(h, w, radius=None):
    """Creates a circular mask of size h x w with given radius centered in the image."""
    center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)



# Apply mask to image (image should already be padded to 442x442)
def apply_mask(image, mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        #print(image.shape, mask.shape, flush = True)
    return image * np.expand_dims(mask, axis=0)  # broadcast mask to channels

def rescale_images(images, device, target_size):
    # rescales the images and normalizes them  
   # Images = np.zeros(((images.shape)[0], 442, 442))
   # Images[:,40:-40, 40:-40] = images
    # Example: size after padding = 442x442
    h, w = 362, 362
    radius = (362 // 2) - 1  # 221
    mask = create_circular_mask(h, w, radius)

    # Convert to torch tensor if needed
    mask_tensor = torch.from_numpy(mask)
    Images = apply_mask(images, mask_tensor)    
    Images = resize(Images, target_size)


    Images = torch.from_numpy(Images).float().to(device)
    for i in range(Images.shape[0]):
        Images[i] = Images[i] - torch.min(Images[i])
        Images[i] = Images[i] / torch.max(Images[i] + 1e-10)
    return Images



def _to_lpips_range_and_rgb(x):
    """
    x: [B,1,H,W] or [B,H,W] in [0,1] (or any float range; we'll clamp)
    returns: [B,3,H,W] in [-1,1]
    """
    if x.ndim == 3:
        x = x.unsqueeze(1)  # [B,1,H,W]
    x = x.float()
    x = torch.clamp(x, 0.0, 1.0)
    x = x.repeat(1, 3, 1, 1)      # grayscale -> 3ch
    x = x * 2.0 - 1.0             # [0,1] -> [-1,1]
    return x

@torch.no_grad()
def compute_lpips(lpips_fn, recos, gts, device):
    """
    recos: [B,1,H,W] or [B,H,W]
    gts  : [B,1,H,W] or [B,H,W]
    """
    recos = _to_lpips_range_and_rgb(recos).to(device)
    gts   = _to_lpips_range_and_rgb(gts).to(device)
    
    return lpips_fn(recos, gts).mean().item()



def augment_image(image, num_augmentations=8):
    augmentation_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
    )

    augmented_images = []
    for _ in range(num_augmentations):
        augmented_image = augmentation_transforms(image)
        augmented_images.append(augmented_image)

    return augmented_images




'>>-------------------------------------------------------------------------<<'
' define functions for computation of validation metrics for s2i_ds'
'>>-------------------------------------------------------------------------<<'


def compute_validation_metrics(full_recos, Ims, batch_size=32):
    """
    Compute SSIM, PSNR, MSE in a memory-safe way.
    Converts tensors to NumPy in small batches to avoid RAM spikes.
    
    full_recos: torch.Tensor, shape [N, C, H, W] or [N, H, W]
    Ims: torch.Tensor, shape [N, H, W]
    """
    N = Ims.shape[0]

    ssim_values, psnr_values, mse_values = [], [], []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Move only the batch to CPU & convert
        full_batch = full_recos[start:end, 0].detach().cpu().numpy().astype(np.float32)
        ims_batch = Ims[start:end].detach().cpu().numpy().astype(np.float32)

        for i in range(full_batch.shape[0]):
            data_range = ims_batch[i].max() - ims_batch[i].min()
            ssim_values.append(structural_similarity(ims_batch[i], full_batch[i], data_range=data_range))
            psnr_values.append(peak_signal_noise_ratio(ims_batch[i], full_batch[i], data_range=data_range))
            mse_values.append(np.mean((ims_batch[i] - full_batch[i]) ** 2))

    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    mean_mse = np.mean(mse_values)

    return mean_ssim, mean_psnr, mean_mse



def compute_validation_metrics_inference(full_recos, Ims):
    
    full_recos_np = full_recos[:, 0].detach().cpu().numpy().astype(np.float32)
    Ims_np = Ims.detach().cpu().numpy().astype(np.float32)
    
    ssim_values, psnr_values, mse_values = [], [], []
    for i in range(len(full_recos_np)):
        data_range = Ims_np[i].max() - Ims_np[i].min()
        ssim_values.append(structural_similarity(Ims_np[i], full_recos_np[i], data_range=data_range))
        psnr_values.append(peak_signal_noise_ratio(Ims_np[i], full_recos_np[i], data_range=data_range))
        mse_values.append(np.mean((Ims_np[i] - full_recos_np[i]) ** 2))  # MSE computation

    return ssim_values, psnr_values, mse_values

def compute_validation_metrics_halfs(even, odd):
    
    even_recos = even[:, 0].detach().cpu().numpy().astype(np.float32)
    odd_recos = odd[:, 0].detach().cpu().numpy().astype(np.float32)
    
    psnr_values = []
    for i in range(len(even_recos)):
        data_range = even_recos[i].max() - odd_recos[i].min()
        psnr_values.append(peak_signal_noise_ratio(even_recos[i], odd_recos[i], data_range=data_range))

    mean_psnr = np.mean(psnr_values)
    
    return mean_psnr





def compute_validation_metrics_S2I_halfs(tensor):
    """
    Compute PSNR across all possible pairs of channels in the tensor.
    
    Args:
        tensor: Tensor of shape [batch, channels, ...]
        
    Returns:
        mean_psnr: float, mean PSNR across all channel pairs and batches
    """
    tensor = tensor.detach().cpu().numpy().astype(np.float32)
    num_channels = tensor.shape[1]
    psnr_values = []

    # Generate all unique channel pairs
    channel_pairs = list(itertools.combinations(range(num_channels), 2))

    for i in range(tensor.shape[0]):  # iterate over batch
        for ch1, ch2 in channel_pairs:
            data_range = tensor[i, ch1].max() - tensor[i, ch2].min()
            psnr_values.append(
                peak_signal_noise_ratio(tensor[i, ch1], tensor[i, ch2], data_range=data_range)
            )

    mean_psnr = np.mean(psnr_values)
    return mean_psnr





def compute_validation_metrics(full_recos, Ims):
    
    full_recos_np = full_recos[:, 0].detach().cpu().numpy().astype(np.float32)
    Ims_np = Ims.detach().cpu().numpy().astype(np.float32)
    
    ssim_values, psnr_values, mse_values = [], [], []
    for i in range(len(full_recos_np)):
        data_range = Ims_np[i].max() - Ims_np[i].min()
        ssim_values.append(structural_similarity(Ims_np[i], full_recos_np[i], data_range=data_range))
        psnr_values.append(peak_signal_noise_ratio(Ims_np[i], full_recos_np[i], data_range=data_range))
        mse_values.append(np.mean((Ims_np[i] - full_recos_np[i]) ** 2))  # MSE computation

    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    mean_mse = np.mean(mse_values)
    
    return mean_ssim, mean_psnr, mean_mse

def compute_validation_metrics_halfs(even, odd):
    
    even_recos = even[:, 0].detach().cpu().numpy().astype(np.float32)
    odd_recos = odd[:, 0].detach().cpu().numpy().astype(np.float32)
    
    psnr_values = []
    for i in range(len(even_recos)):
        data_range = even_recos[i].max() - odd_recos[i].min()
        psnr_values.append(peak_signal_noise_ratio(even_recos[i], odd_recos[i], data_range=data_range))

    mean_psnr = np.mean(psnr_values)
    
    return mean_psnr




#-------------------------- Random2inverse -------------------------------#
    
class Proj2Proj:
    def __init__(self, device="cuda:0", random = True, grid_size = 3, fill_zeros = True):
        self.device = device
        self.folds = 1
        self.batch_size = 8
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.grid_size = grid_size
        self.conv_local_avg = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding="same", padding_mode="reflect")
        self.conv_local_avg.requires_grad_(False)
        self.conv_local_avg.weight[...] = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=torch.float32)
        self.random = random
        self.fill_zeros = fill_zeros
        
        
    def forward(self, reconstruction, nr_angles, invariant_inference = False):
        output_denoising = self.net_denoising(reconstruction.float().to(self.device)) 
        if invariant_inference == False:
           # mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
           # mask_tensor = torch.from_numpy(mask)
        #    output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino = self.projection_tomosipo(output_denoising, nr_angles)
        else:
            mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
            mask_tensor = torch.from_numpy(mask)
            output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino, angles, angles_old = self.projection_tomosipo(output_denoising, nr_angles, invariant_inference = True)
        
        if invariant_inference == False:
            return output_denoising, output_denoising_sino
        else:
            return output_denoising, output_denoising_sino, angles, angles_old


    def prepare_batch(self, sinograms, iteration):

        B, C, H, W = sinograms.shape  # (batch, 1, 336, 64)

        # Pick grid size
        if isinstance(self.grid_size, Number):
            grid_size = self.grid_size
        else:
            grid_size = random.choice(self.grid_size)

        phasex = iteration % grid_size
        phasey = (iteration // grid_size) % grid_size

        # Create mask ON GPU directly
        if not self.random:
            sinogram_mask = self.pixel_grid_mask(
                sinograms[0].shape, grid_size, phasex, phasey
            ).to(sinograms.device)
        else:
            sinogram_mask = self.random_safe_mask(
                sinograms.shape, grid_size, sinograms.device
            )
        sinogram_mask_c = 1 - sinogram_mask

        # Masked sinogram
        if self.fill_zeros:
            masked = self.interpolate_mask_new(
                sinograms, sinogram_mask, sinogram_mask_c, iteration
            )
        else:
            masked = sinograms * sinogram_mask_c

        # Compute reconstructions in batch
        # MOVE EVERYTHING TO CPU ONLY IF FBP REQUIRES IT
        masked_cpu = masked[:, 0].detach().cpu()  # (B, H, W)


        theta = np.linspace(0.0, np.pi, W, endpoint=False)

        device = sinograms.device

        # preallocate GPU tensor
        reconstructions = torch.empty(
            (B, 1, H, H), device=device, dtype=sinograms.dtype
        )

        # batch FBP WITHOUT CPU HOPS
        for i in range(B):
            reconstructions[i] = self.fbp_tomosipo(
                masked[i:i+1], angle_vector=theta, folds=self.folds
            )

        return reconstructions, sinograms, sinogram_mask.unsqueeze(0)



            
    def random_safe_3x3_mask(self, shape, grid_size, device):
        """
        shape = (B, C, H, W) or (H, W)
        Returns a mask with:
        - exactly 1 pixel per 3x3 block
        - no 4-connected neighbors across blocks
        """

        if len(shape) == 4:
            H, W = shape[-2:]
        else:
            H, W = shape

        gh = H // grid_size
        gw = W // grid_size

        mask = torch.zeros((H, W), device=device)

        # store chosen offsets per block
        chosen = torch.empty((gh, gw, 2), dtype=torch.long, device=device)

        all_offsets = torch.tensor(
            [(y, x) for y in range(grid_size) for x in range(grid_size)],
            device=device
        )

        for i in range(gh):
            for j in range(gw):

                valid = []

                for off in all_offsets:
                    ok = True

                    # check block above
                    if i > 0:
                        up = chosen[i - 1, j]
                        if off[0] == 0 and up[0] == 2 and off[1] == up[1]:
                            ok = False

                    # check block to the left
                    if j > 0:
                        left = chosen[i, j - 1]
                        if off[1] == 0 and left[1] == 2 and off[0] == left[0]:
                            ok = False

                    if ok:
                        valid.append(off)

                valid = torch.stack(valid)
                idx = torch.randint(len(valid), (1,), device=device)
                chosen[i, j] = valid[idx]

                y = i * grid_size + chosen[i, j, 0]
                x = j * grid_size + chosen[i, j, 1]
                mask[y, x] = 1.0

        return mask
            
    def random_safe_mask(self, shape, grid_size, device):
        """
        shape = (B, C, H, W) or (H, W)
        Returns a mask with:
        - exactly 1 pixel per grid_size×grid_size block
        - no 4-connected neighbors globally
        """

        if len(shape) == 4:
            H, W = shape[-2:]
        else:
            H, W = shape

        gh = H // grid_size
        gw = W // grid_size

        mask = torch.zeros((H, W), device=device)

        # store chosen GLOBAL coordinates per block
        chosen_y = torch.full((gh, gw), -1, dtype=torch.long, device=device)
        chosen_x = torch.full((gh, gw), -1, dtype=torch.long, device=device)

        offsets = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij"
        ), dim=-1).reshape(-1, 2)

        for i in range(gh):
            for j in range(gw):

                base_y = i * grid_size
                base_x = j * grid_size

                # global coords of all candidates in this block
                cand_y = base_y + offsets[:, 0]
                cand_x = base_x + offsets[:, 1]

                valid = torch.ones(len(offsets), dtype=torch.bool, device=device)

                # check block above
                if i > 0:
                    uy = chosen_y[i-1, j]
                    ux = chosen_x[i-1, j]
                    valid &= ~((cand_y == uy + 1) & (cand_x == ux))

                # check block left
                if j > 0:
                    ly = chosen_y[i, j-1]
                    lx = chosen_x[i, j-1]
                    valid &= ~((cand_y == ly) & (cand_x == lx + 1))

                # pick random valid
                ids = torch.nonzero(valid).squeeze(1)
                k = ids[torch.randint(len(ids), (1,), device=device)]

                y = cand_y[k]
                x = cand_x[k]

                chosen_y[i, j] = y
                chosen_x[i, j] = x
                mask[y, x] = 1.0

        return mask

    def prepare_batch_test(self, sinograms):
        #sinograms = sinograms.squeeze()

        reconstructions = np.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2])
        )
        number_of_angles = sinograms.shape[-1]
            
        theta = np.linspace(0.0, np.pi, number_of_angles, endpoint=False)

        
        for i in range(sinograms.shape[0]):
            for j in range(1):
                I = sinograms[i,j].cpu()
                ### input of fbp should have shape [1,1, s, theta]
                reconstructions[i, j] = self.fbp_tomosipo(
                    torch.tensor(I.unsqueeze(0).unsqueeze(0)),
                    angle_vector=theta,
                    folds=1,
                )

        return (
            torch.tensor(reconstructions),
            sinograms,
        )

    def projection_tomosipo(self, img, sino, invariant_inference=False):
        """Compute tomographic projection."""
        angles = sino if isinstance(sino, int) else sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336)
        )
        if invariant_inference == False:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)
        else:
            geo.angles=np.linspace(0, np.pi, 544, endpoint=False)
            angles_old = np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)

        if invariant_inference == False:
            return torch.moveaxis(sino, -1, -2)
        else:
            return torch.moveaxis(sino, -1, -2), angles_old, geo.angles

    def fbp_tomosipo(self, sino, angle_vector=None, folds=None):
        """Perform filtered back-projection reconstruction."""
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
        )
        
        if angle_vector is not None and angle_vector[0] is not None:
            geo.angles = angle_vector
        else:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False)
        
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        return fbp(op, sino[:, 0]).unsqueeze(1)



    def pixel_grid_mask(self, shape, patch_size, phase_x, phase_y):
        A = torch.zeros(shape[-2:])
        for i in range(shape[-2]):
            for j in range(shape[-1]):
                if (i % patch_size == phase_x and j % patch_size == phase_y):
                    A[i, j] = 1
        return torch.Tensor(A)



    def interpolate_mask_new(self, tensor, mask, mask_inv, iteration):
        device = tensor.device
        mask = mask.to(device)
        mask_inv = mask_inv.to(device)


        proj_in_copy = torch.clone(tensor)
        #filtered_tensor = self.conv_local_avg(tensor)
        self.conv_local_avg = self.conv_local_avg.to(tensor.device)
        filtered_tensor = self.conv_local_avg(tensor)

        # --- combine: update masked pixels only ---
        combined = filtered_tensor * mask + proj_in_copy * mask_inv

        return combined






#-------------------------Sparse2inverse splitting -------------------------------#
    
class Sparse2Inverse_p2p:
    def __init__(self, device="cuda:0", random = False, grid_size=3, fill_zeros = True):
        self.device = device
        self.folds = 1
        self.batch_size = 8
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.grid_size = grid_size
        self.random = random
        self.interpolate = fill_zeros
        self.conv_local_avg = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1, 3),bias=False,padding=(0, 1),padding_mode="reflect")
        self.conv_local_avg.requires_grad_(False)
        self.conv_local_avg.weight[...] =   torch.tensor(
        [[[[0.5, 0.0, 0.5]]]], dtype=torch.float32)


    def forward(self, reconstruction, nr_angles, invariant_inference = False):
        output_denoising = self.net_denoising(reconstruction.float().to(self.device)) 
        if invariant_inference == False:
           # mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
           # mask_tensor = torch.from_numpy(mask)
        #    output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino = self.projection_tomosipo(output_denoising, nr_angles)
        else:
            mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
            mask_tensor = torch.from_numpy(mask)
            output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino, angles, angles_old = self.projection_tomosipo(output_denoising, nr_angles, invariant_inference = True)
        
        if invariant_inference == False:
            return output_denoising, output_denoising_sino
        else:
            return output_denoising, output_denoising_sino, angles, angles_old



    def prepare_batch(self, sinograms, iteration):

        reconstructions = np.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2])
        )
        number_of_angles = sinograms.shape[-1]

        if isinstance(self.grid_size, Number):
            grid_size = self.grid_size
        elif isinstance(self.grid_size, (list, tuple)):
            grid_size = random.choice(self.grid_size)
        else:
            raise TypeError("grid_size must be a number or a list/tuple of numbers")
        ## in case grid size = 3, we have 9 partitions, so we pick 1 out of 9 angles for each partition, thus G = grid_size^2
        G = grid_size ** 2


        gen = torch.Generator(device='cpu')
        gen.seed()

        # ==========================
        # ANGULAR SPLIT
        # ==========================
        if self.random:
            n_blocks = (number_of_angles + G - 1) // G
            idx_angles = []
            prev_choice = None

            for b in range(n_blocks):
                start = b * G
                end = min((b + 1) * G, number_of_angles)
                choices = torch.arange(start, end)

                if prev_choice is not None:
                    forbidden = torch.tensor(
                        [prev_choice - 1, prev_choice, prev_choice + 1]
                    )
                    choices = choices[~torch.isin(choices, forbidden)]

                # safety check
                if len(choices) == 0:
                    raise RuntimeError("No valid angles left after neighbor exclusion")
                # safe CPU generator usage
                idx = torch.randint(len(choices), (1,), generator=gen)  # CPU
                choice = choices[idx.to(choices.device)]                # convert

               # choice = choices[torch.randint(len(choices), (1,), generator=gen)]
                idx_angles.append(choice.item())
                prev_choice = choice.item()

        else:
            index_angle = iteration % G
            idx_angles = []
            while index_angle < number_of_angles:
                idx_angles.append(index_angle)
                index_angle += G

        sinogram_mask = torch.zeros(
            sinograms.shape[-2], sinograms.shape[-1], device=sinograms.device
        )
        sinogram_mask[:, idx_angles] = 1.0
        sinogram_mask_c = 1.0 - sinogram_mask

        sinogram_1 = sinogram_mask_c * sinograms

        theta = np.linspace(0.0, np.pi, number_of_angles, endpoint=False)

        if self.interpolate:
            sinogram_1 = self.interpolate_mask_new(
                sinograms, sinogram_mask, sinogram_mask_c, iteration
            )

        masks = sinogram_mask.cpu()

        # ==========================
        # FBP reconstruction
        # ==========================
        for i in range(sinogram_1.shape[0]):
            I = sinogram_1[i, 0].cpu()
            reconstructions[i, 0] = self.fbp_tomosipo(
                torch.tensor(I.unsqueeze(0).unsqueeze(0)),
                angle_vector=theta,
                folds=self.folds,
            )

        return (
            torch.tensor(reconstructions),
            sinograms,
            masks.unsqueeze(0)
        )


    def prepare_batch_test(self, sinograms):
        #sinograms = sinograms.squeeze()

        reconstructions = np.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2])
        )
        number_of_angles = sinograms.shape[-1]
            
        theta = np.linspace(0.0, np.pi, number_of_angles, endpoint=False)

        
        for i in range(sinograms.shape[0]):
            for j in range(1):
                I = sinograms[i,j].cpu()
                ### input of fbp should have shape [1,1, s, theta]
                reconstructions[i, j] = self.fbp_tomosipo(
                    torch.tensor(I.unsqueeze(0).unsqueeze(0)),
                    angle_vector=theta,
                    folds=1,
                )

        return (
            torch.tensor(reconstructions),
            sinograms,
        )

    def projection_tomosipo(self, img, sino, invariant_inference=False):
        """Compute tomographic projection."""
        angles = sino if isinstance(sino, int) else sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336)
        )
        if invariant_inference == False:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)
        else:
            geo.angles=np.linspace(0, np.pi, 544, endpoint=False)
            angles_old = np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)

        if invariant_inference == False:
            return torch.moveaxis(sino, -1, -2)
        else:
            return torch.moveaxis(sino, -1, -2), angles_old, geo.angles

    def fbp_tomosipo(self, sino, angle_vector=None, folds=None):
        """Perform filtered back-projection reconstruction."""
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
        )     
        if angle_vector is not None and angle_vector[0] is not None:
            geo.angles = angle_vector
        else:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False) 
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        return fbp(op, sino[:, 0]).unsqueeze(1)


    def interpolate_mask_new(self, tensor, mask, mask_inv, iteration):
        device = tensor.device
        self.conv_local_avg = self.conv_local_avg.to(tensor.device)
        mask = mask.to(device)
        mask_inv = mask_inv.to(device)
        proj_in_copy = torch.clone(tensor)
        filtered_tensor = self.conv_local_avg(tensor)
        # --- combine: update masked pixels only ---
        combined = filtered_tensor * mask + proj_in_copy * mask_inv
        return combined




    

#-------------------------Sparse2inverse doublesplit splitting -------------------------------#
    
class Sparse2Inverse_ds_p2p:
    def __init__(self, device="cuda:0", random = False, grid_size=3, fill_zeros = True):
        self.device = device
        self.folds = 1
        self.batch_size = 8
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.grid_size = grid_size
        self.random = random
        self.interpolate = fill_zeros
        # θ-direction interpolation (along angles)
        


        self.conv_theta = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=False,
            padding_mode="reflect",
        )

        # s-direction interpolation (along detector bins)
        self.conv_s = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
            padding_mode="reflect",
        )
        self.conv_theta = self.conv_theta.to(self.device)
        self.conv_s     = self.conv_s.to(self.device)
        
        
        for conv in [self.conv_theta, self.conv_s]:
            conv.requires_grad_(False)

        with torch.no_grad():
            self.conv_theta.weight[:] = torch.tensor([[[[0.5, 0.0, 0.5]]]])
            self.conv_s.weight[:] = torch.tensor([[[[0.5], [0.0], [0.5]]]])


    def forward(self, reconstruction, nr_angles, invariant_inference = False):
        output_denoising = self.net_denoising(reconstruction.float().to(self.device))
        if invariant_inference == False:
            output_denoising_sino = self.projection_tomosipo(output_denoising, nr_angles)
        else:
            mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
            mask_tensor = torch.from_numpy(mask)
            output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino, angles, angles_old = self.projection_tomosipo(output_denoising, nr_angles, invariant_inference = True)
        
        if invariant_inference == False:
            return output_denoising, output_denoising_sino
        else:
            return output_denoising, output_denoising_sino, angles, angles_old

    
    def prepare_batch(self, sinograms, iteration):
        sinograms = sinograms.to(self.device)
        B = sinograms.shape[0]
        S = sinograms.shape[-2]
        T = sinograms.shape[-1]

        reco_theta = np.zeros((B, 1, S, S))
        reco_s = np.zeros((B, 1, S, S))
        target_reco = np.zeros((B, 1, S, S))
    
        # independent RNG (not affected by global torch.manual_seed)
        gen = torch.Generator()        
        gen.seed()
        if isinstance(self.grid_size, Number):
            grid_size = self.grid_size
        elif isinstance(self.grid_size, (list, tuple)):
            grid_size = random.choice(self.grid_size)
        else:
            raise TypeError("grid_size must be a number or a list/tuple of numbers")

        G = grid_size ** 2
        # =============================
        # ANGULAR (theta) SPLIT
        # =============================
        if self.random:
            n_blocks = (T + G - 1) // G
            idx_angles = []
            prev_choice = None

            for b in range(n_blocks):
                start = b * G
                end = min((b + 1) * G, T)
                choices = torch.arange(start, end, device=sinograms.device)

                if prev_choice is not None:
                    forbidden = torch.tensor(
                        [prev_choice - 1, prev_choice, prev_choice + 1],
                    )
                    choices = choices[~torch.isin(choices, forbidden.to(sinograms.device))]

                if len(choices) == 0:
                    raise RuntimeError("No valid angles left after neighbor exclusion")

                idx = torch.randint(len(choices), (1,), generator=gen)
                choice = choices[idx.to(choices.device)]

                idx_angles.append(choice.item())
                prev_choice = choice.item()

        else:
            index_angle = iteration % (G)
            idx_angles = []
            while index_angle < T:
                idx_angles.append(index_angle)
                index_angle += G 
    
        mask_theta = torch.zeros(S, T, device=sinograms.device)
        mask_theta[:, idx_angles] = 1.0
        mask_theta_c = 1.0 - mask_theta
    
        sino_theta = mask_theta_c * sinograms
        if self.interpolate:
            sino_theta = self.interpolate_mask_new(
                sinograms, mask_theta, mask_theta_c,
                iteration, direction="theta"
            )
    
        # =============================
        # DETECTOR (s) SPLIT
        # =============================

        if self.random:
            n_blocks = (S + G - 1) // G
            idx_s = []
            prev_choice = choice.item()


            for b in range(n_blocks):
                start = b * G
                end = min((b + 1) * G, S)
                choices = torch.arange(start, end, device=sinograms.device)

                if prev_choice is not None:
                    forbidden = torch.tensor(
                        [prev_choice - 1, prev_choice, prev_choice + 1],
                    )
                    choices = choices[~torch.isin(choices, forbidden.to(sinograms.device))]

                if len(choices) == 0:
                    raise RuntimeError("No valid detector rows left after neighbor exclusion")

                idx = torch.randint(len(choices), (1,), generator=gen)
                choice = choices[idx.to(choices.device)]
                idx_s.append(choice.item())
                prev_choice = choice.item()

        else:
            index_s = iteration % (G)
            idx_s = []
            while index_s < S:
                idx_s.append(index_s)
                index_s += G 
    
        mask_s = torch.zeros(S, T, device=sinograms.device)
        mask_s[idx_s, :] = 1.0
        mask_s_c = 1.0 - mask_s
    
        sino_s = mask_s_c * sinograms
        if self.interpolate:
            sino_s = self.interpolate_mask_new(
                sinograms, mask_s, mask_s_c,
                iteration, direction="s"
            )
    
        # =============================
        # FBP reconstructions
        # =============================
        # --- before the loop (CPU theta vector too) ---
        theta = np.linspace(0.0, np.pi, T, endpoint=False)

        with torch.no_grad():
            for i in range(B):
                reco_theta[i, 0] = self.fbp_tomosipo(
                    sino_theta[i:i+1].detach().cpu().contiguous(),
                    angle_vector=theta,
                    folds=self.folds,
                )

                reco_s[i, 0] = self.fbp_tomosipo(
                    sino_s[i:i+1].detach().cpu().contiguous(),
                    angle_vector=theta,
                    folds=self.folds,
                )

                target_reco[i, 0] = self.fbp_tomosipo(
                    sinograms[i:i+1].detach().cpu().contiguous(),
                    angle_vector=theta,
                    folds=self.folds,
                )
    
        return (
            torch.from_numpy(reco_theta).float(),
            torch.from_numpy(reco_s).float(),
            torch.from_numpy(target_reco).float(),
            sinograms,
            mask_theta.unsqueeze(0).cpu(),
            mask_s.unsqueeze(0).cpu(),
        )

    

    def prepare_batch_test(self, sinograms):
        #sinograms = sinograms.squeeze()

        reconstructions = np.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2])
        )
        number_of_angles = sinograms.shape[-1]
            
        theta = np.linspace(0.0, np.pi, number_of_angles, endpoint=False)

        
        for i in range(sinograms.shape[0]):
            for j in range(1):
                I = sinograms[i,j].cpu()
                ### input of fbp should have shape [1,1, s, theta]
                reconstructions[i, j] = self.fbp_tomosipo(
                    torch.tensor(I.unsqueeze(0).unsqueeze(0)),
                    angle_vector=theta,
                    folds=1,
                )

        return (
            torch.tensor(reconstructions),
            sinograms,
        )

    def projection_tomosipo(self, img, sino, invariant_inference=False):
        """Compute tomographic projection."""
        angles = sino if isinstance(sino, int) else sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336)
        )
        if invariant_inference == False:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)
        else:
            geo.angles=np.linspace(0, np.pi, 544, endpoint=False)
            angles_old = np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)

        if invariant_inference == False:
            return torch.moveaxis(sino, -1, -2)
        else:
            return torch.moveaxis(sino, -1, -2), angles_old, geo.angles
            

    def fbp_tomosipo(self, sino, angle_vector=None, folds=None):
        """Perform filtered back-projection reconstruction."""
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
        )     
        if angle_vector is not None and angle_vector[0] is not None:
            geo.angles = angle_vector
        else:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False) 
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        return fbp(op, sino[:, 0]).unsqueeze(1)


    def interpolate_mask_new(self, tensor, mask, mask_inv, iteration, direction):
        """
        tensor    : [B,1,S,T]
        mask      : [S,T]   (1 = masked → interpolate)
        mask_inv  : [S,T]   (1 = keep original)
        direction : "theta" or "s"
        """

        device = tensor.device

        # Broadcast masks
        mask = mask.to(device).unsqueeze(0).unsqueeze(0)       # [1,1,S,T]
        mask_inv = mask_inv.to(device).unsqueeze(0).unsqueeze(0)

        original = tensor  # no clone
        # -------------------------------
        # Detect mask orientation
        # -------------------------------
        col_variance = mask[0, 0].sum(dim=0).var()   # θ variation
        row_variance = mask[0, 0].sum(dim=1).var()   # s variation



        if col_variance > row_variance:
            filtered = self.conv_theta(tensor)  # angular interpolation
            interp_dir = "theta"
        else:
            filtered = self.conv_s(tensor)      # s-direction interpolation
            interp_dir = "s"

        # Replace masked pixels only
        combined = filtered * mask + original * mask_inv



        return combined








class Sparse2Inverse_ds_all_combinations:
    def __init__(self, device="cuda:0", random = False, grid_size=3, fill_zeros = True):
        self.device = device
        self.folds = 1
        self.batch_size = 8
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.grid_size = grid_size
        self.random = random
        self.interpolate = fill_zeros
        # θ-direction interpolation (along angles)
        


        self.conv_theta = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=False,
            padding_mode="reflect",
        )

        # s-direction interpolation (along detector bins)
        self.conv_s = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
            padding_mode="reflect",
        )
        self.conv_theta = self.conv_theta.to(self.device)
        self.conv_s     = self.conv_s.to(self.device)
        
        
        for conv in [self.conv_theta, self.conv_s]:
            conv.requires_grad_(False)

        with torch.no_grad():
            self.conv_theta.weight[:] = torch.tensor([[[[0.5, 0.0, 0.5]]]])
            self.conv_s.weight[:] = torch.tensor([[[[0.5], [0.0], [0.5]]]])


    def forward(self, reconstruction, nr_angles, invariant_inference = False):
        output_denoising = self.net_denoising(reconstruction.float().to(self.device))
        if invariant_inference == False:
            output_denoising_sino = self.projection_tomosipo(output_denoising, nr_angles)
        else:
            mask = create_circular_mask( output_denoising.shape[-2],  output_denoising.shape[-2], output_denoising.shape[-2]//2)
            # Convert to torch tensor if needed
            mask_tensor = torch.from_numpy(mask)
            output_denoising = apply_mask(output_denoising.detach().cpu(), mask_tensor).to(self.device)
            output_denoising_sino, angles, angles_old = self.projection_tomosipo(output_denoising, nr_angles, invariant_inference = True)
        
        if invariant_inference == False:
            return output_denoising, output_denoising_sino
        else:
            return output_denoising, output_denoising_sino, angles, angles_old

    



    def prepare_batch(self, sinograms, iteration_theta, iteration_s):

        sinograms = sinograms.to(self.device)
        B = sinograms.shape[0]
        S = sinograms.shape[-2]
        T = sinograms.shape[-1]


        reco_theta = np.zeros((B, 1, S, S))
        reco_s = np.zeros((B, 1, S, S))
        target_reco = np.zeros((B, 1, S, S))

        reconstructions = np.zeros((B, 1, S, S))
        number_of_angles = T

        # --------------------------
        # grid size handling
        # --------------------------
        if isinstance(self.grid_size, Number):
            grid_size = self.grid_size
        elif isinstance(self.grid_size, (list, tuple)):
            grid_size = random.choice(self.grid_size)
        else:
            raise TypeError("grid_size must be number or list/tuple")

        G = grid_size ** 2   

        # =========================================================
        # DOUBLE SPLIT INDEXING (FULL COMBINATIONS)
        # =========================================================
        angle_id = iteration_theta % G
        det_id   = iteration_s % G



        # =========================================================
        # ANGULAR MASK
        # =========================================================
        idx_angles = []
        start_a = angle_id
        while start_a < number_of_angles:
            idx_angles.append(start_a)
            start_a += G

        sinogram_mask = torch.zeros(S, T, device=sinograms.device)
        sinogram_mask[:, idx_angles] = 1.0

        sinogram_mask_c = 1.0 - sinogram_mask
        sino_theta = sinogram_mask_c * sinograms

        # =========================================================
        # DETECTOR MASK
        # =========================================================
        idx_s = []
        start_s = det_id
        while start_s < S:
            idx_s.append(start_s)
            start_s += G

        sinogram_mask_s = torch.zeros(S, T, device=sinograms.device)
        sinogram_mask_s[idx_s, :] = 1.0

        sinogram_mask_s_c = 1.0 - sinogram_mask_s
        sino_s = sinogram_mask_s_c * sinograms

        # =========================================================
        # OPTIONAL INTERPOLATION (kept consistent)
        # =========================================================
        if self.interpolate:
            sino_theta = self.interpolate_mask_new(
                sinograms, sinogram_mask, sinogram_mask_c, iteration_theta, direction="theta"
            )
            sino_s = self.interpolate_mask_new(
                sinograms, sinogram_mask_s, sinogram_mask_s_c, iteration_s, direction="s"
            )

        # =========================================================
        # FBP
        # =========================================================
        theta = np.linspace(0.0, np.pi, T, endpoint=False)

        for i in range(B):
            reco_theta[i, 0] = self.fbp_tomosipo(
                sino_theta[i:i+1].detach().cpu().contiguous(),
                angle_vector=theta,
                folds=self.folds,
            )

            reco_s[i, 0] = self.fbp_tomosipo(
                sino_s[i:i+1].detach().cpu().contiguous(),
                angle_vector=theta,
                folds=self.folds,
            )

        # target (full)
        target_reco = np.zeros((B, 1, S, S))
        for i in range(B):
            target_reco[i, 0] = self.fbp_tomosipo(
                sinograms[i:i+1].detach().cpu().contiguous(),
                angle_vector=theta,
                folds=self.folds,
            )

        return (
            torch.from_numpy(reco_theta).float(),
            torch.from_numpy(reco_s).float(),
            torch.from_numpy(target_reco).float(),
            sinograms,
            sinogram_mask.unsqueeze(0).cpu(),
            sinogram_mask_s.unsqueeze(0).cpu(),
        )
        

    def prepare_batch_test(self, sinograms):
        #sinograms = sinograms.squeeze()

        reconstructions = np.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2])
        )
        number_of_angles = sinograms.shape[-1]
            
        theta = np.linspace(0.0, np.pi, number_of_angles, endpoint=False)

        
        for i in range(sinograms.shape[0]):
            for j in range(1):
                I = sinograms[i,j].cpu()
                ### input of fbp should have shape [1,1, s, theta]
                reconstructions[i, j] = self.fbp_tomosipo(
                    torch.tensor(I.unsqueeze(0).unsqueeze(0)),
                    angle_vector=theta,
                    folds=1,
                )

        return (
            torch.tensor(reconstructions),
            sinograms,
        )

    def projection_tomosipo(self, img, sino, invariant_inference=False):
        """Compute tomographic projection."""
        angles = sino if isinstance(sino, int) else sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336)
        )
        if invariant_inference == False:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)
        else:
            geo.angles=np.linspace(0, np.pi, 544, endpoint=False)
            angles_old = np.linspace(0, np.pi, angles, endpoint=False)
            op = to_autograd(ct.make_operator(geo))
            sino = op(img[:, 0].to(self.device)).unsqueeze(1)

        if invariant_inference == False:
            return torch.moveaxis(sino, -1, -2)
        else:
            return torch.moveaxis(sino, -1, -2), angles_old, geo.angles
            

    def fbp_tomosipo(self, sino, angle_vector=None, folds=None):
        """Perform filtered back-projection reconstruction."""
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
        )     
        if angle_vector is not None and angle_vector[0] is not None:
            geo.angles = angle_vector
        else:
            geo.angles=np.linspace(0, np.pi, angles, endpoint=False) 
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        return fbp(op, sino[:, 0]).unsqueeze(1)


    def interpolate_mask_new(self, tensor, mask, mask_inv, iteration, direction):
        """
        tensor    : [B,1,S,T]
        mask      : [S,T]   (1 = masked → interpolate)
        mask_inv  : [S,T]   (1 = keep original)
        direction : "theta" or "s"
        """

        device = tensor.device

        # Broadcast masks
        mask = mask.to(device).unsqueeze(0).unsqueeze(0)       # [1,1,S,T]
        mask_inv = mask_inv.to(device).unsqueeze(0).unsqueeze(0)

        original = tensor  # no clone
        # -------------------------------
        # Detect mask orientation
        # -------------------------------
        col_variance = mask[0, 0].sum(dim=0).var()   # θ variation
        row_variance = mask[0, 0].sum(dim=1).var()   # s variation



        if col_variance > row_variance:
            filtered = self.conv_theta(tensor)  # angular interpolation
            interp_dir = "theta"
        else:
            filtered = self.conv_s(tensor)      # s-direction interpolation
            interp_dir = "s"

        # Replace masked pixels only
        combined = filtered * mask + original * mask_inv



        return combined





def validate_average(validation_dataloader, N2I, random=False):
    full_recos = []
    MSEs = []
    Ims = []
    Recos = []
    random_partitioning = N2I.random
    N2I.random = False  # ensure deterministic partitioning during validation

    # normalize grid_size
    grid_sizes = N2I.grid_size
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes]

    with torch.no_grad():
        for sinos, ims in validation_dataloader:
            sinos = sinos.to(N2I.device)
            ims = ims.to(N2I.device)

            recos_given = N2I.fbp_tomosipo(sinos)

            batch_size = sinos.shape[0]
            H, W = ims.shape[-2], ims.shape[-1]
            num_angles = sinos.shape[-1]

            final_recos_all_grids = []

            # ---------- LOOP OVER GRID SIZES ----------
            for g in grid_sizes:
                N2I.grid_size = g        # override here

                num_splits = g ** 2
                assert num_angles >= num_splits

                final_recos = torch.zeros(
                    (batch_size, 1, H, W), device=N2I.device
                )

                # ---------- LOOP OVER PARTITIONS ----------
                for iteration in range(num_splits):
                    recos, sinos_masked, _ = N2I.prepare_batch(
                        sinos.cpu(), iteration
                    )

                    recos = recos.to(N2I.device)
                    sinos_masked = sinos_masked.to(N2I.device)

                    input_x_den = recos[:, 0:1]
                    output_reco, _ = N2I.forward(input_x_den, sinos_masked)
                    
                    mask = create_circular_mask( output_reco.shape[-2],  output_reco.shape[-2], output_reco.shape[-2]//2)
                    # Convert to torch tensor if needed
                    mask_tensor = torch.from_numpy(mask)
                    output_reco = apply_mask(output_reco.detach().cpu(), mask_tensor).to(N2I.device)

                    final_recos += output_reco / num_splits
                    final_recos_all_grids.append(final_recos)

            # ---------- AVERAGE OVER GRID SIZES ----------
            final_recos = torch.stack(final_recos_all_grids, dim=0).mean(dim=0)
            full_recos.append(final_recos)
            Ims.append(ims)
            Recos.append(recos_given)

            err = torch.mean(
                torch.mean((final_recos.squeeze(1) - ims) ** 2, dim=-1),
                dim=-1,
            )
            MSEs.append(err)

    # restore original grid_size
    N2I.grid_size = grid_sizes if len(grid_sizes) > 1 else grid_sizes[0]
    N2I.random = random_partitioning  # restore original random setting
    full_recos = torch.cat(full_recos, dim=0)
    Ims = torch.cat(Ims, dim=0)
    
    Recos = torch.cat(Recos, dim=0)
    MSEs = torch.cat(MSEs, dim=0)


    return full_recos, MSEs, Ims, Recos






def validate_P_invariant(validation_dataloader, N2I, random=False, full_size = 544):
    full_recos = []
    MSEs = []
    Ims = []
    Recos = []
    random_partitioning = N2I.random
    N2I.random = False  # ensure deterministic partitioning during validation

    grid_sizes = N2I.grid_size
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes]

    plotted_masked = False
    plotted_final = False

    with torch.no_grad():
        for sinos, ims in validation_dataloader:

            sinos = sinos.to(N2I.device)
            ims = ims.to(N2I.device)

            recos_given = N2I.fbp_tomosipo(sinos)



            batch_size = sinos.shape[0]
            H, W = ims.shape[-2], ims.shape[-1]
            num_angles = sinos.shape[-1]

            final_recos_all_grids = []

            # ---------- LOOP OVER GRID SIZES ----------
            for g in grid_sizes:

                N2I.grid_size = g
                num_splits = g ** 2
                assert num_angles >= num_splits

                final_sino = None
                mask_sum = torch.zeros((sinos.shape[0],1,336,full_size)).to(sinos.device)

                # ---------- LOOP OVER PARTITIONS ----------
                for iteration in range(num_splits):
                    recos, sinos_masked, mask = N2I.prepare_batch(
                        sinos.cpu(), iteration
                    )
                    recos = recos.to(N2I.device)
                    sinos_masked = sinos_masked.to(N2I.device)

                    input_x_den = recos[:, 0:1]
                    _, output_sino, ang, old_ang = N2I.forward(input_x_den, sinos_masked, invariant_inference = True)
      
                  
                  # Assuming mask has shape [1, H, W_orig] and we want W_new = 544
                    B, H, W_orig = mask.shape
                    W_new = 544
                    expand_factor = W_new // W_orig  # e.g., 544 / 16 = 34

                # Step 1: repeat each original column expand_factor times
                    mask_big = torch.ones(1,336,full_size)  # [B, H, W_new]
                    for i in range(W_orig):
                        ### we now set the values zero that were used to create the prediction
                        mask_big[:,:,i*expand_factor] = mask[:,:,i]

    
                    mask = mask_big.to(output_sino.device)
                   # print(mask.shape, output_sino.shape, flush = True)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                    #mask = mask.expand_as(output_sino)
                    ### we now mask the sinogram prediction to only keep the values that were predicted without the ones used for prediction
                    output_sino_J = output_sino * mask

                    if final_sino is None:
                        final_sino = torch.zeros_like(output_sino)

                    final_sino += output_sino_J
                    mask_sum += mask

                final_reco = N2I.fbp_tomosipo(final_sino/mask_sum)
                
                mask = create_circular_mask( final_reco.shape[-2],  final_reco.shape[-2], final_reco.shape[-2]//2)
                mask_tensor = torch.from_numpy(mask)
                final_reco = apply_mask(final_reco.detach().cpu(), mask_tensor).to(N2I.device)
                
                final_recos_all_grids.append(final_reco)
                
           # print("Mask sum unique values:", torch.unique(mask_sum), flush=True)
            
            final_recos = torch.stack(final_recos_all_grids, dim=0).mean(dim=0)

            full_recos.append(final_recos)
            Ims.append(ims)
            Recos.append(recos_given)

            err = torch.mean(
                torch.mean((final_recos.squeeze(1) - ims) ** 2, dim=-1),
                dim=-1,
            )
            MSEs.append(err)

    N2I.grid_size = grid_sizes if len(grid_sizes) > 1 else grid_sizes[0]
    N2I.random = random_partitioning
    full_recos = torch.cat(full_recos, dim=0)
    Ims = torch.cat(Ims, dim=0)
    Recos = torch.cat(Recos, dim=0)
    MSEs = torch.cat(MSEs, dim=0)

    return full_recos, MSEs, Ims, Recos



def validate_P_invariant_doublesplit(validation_dataloader, N2I, random=False, full_size = 544):
    full_recos = []
    MSEs = []
    Ims = []
    Recos = []

    grid_sizes = N2I.grid_size
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes]

    plotted_masked = False
    plotted_final = False

    with torch.no_grad():
        for sinos, ims in validation_dataloader:
            
            sinos = sinos.to(N2I.device)
            ims = ims.to(N2I.device)
            recos_given = N2I.fbp_tomosipo(sinos)

            batch_size = sinos.shape[0]
            H, W = ims.shape[-2], ims.shape[-1]
            num_angles = sinos.shape[-1]

            final_recos_all_grids = []

            for g in grid_sizes:
                N2I.grid_size = g        # override here

                num_splits = g ** 2

                final_sino = None
                mask_sum = torch.zeros((sinos.shape[0],1,336,full_size)).to(sinos.device)
                # =====================================
                # loop over Sparse2Inverse partitions
                # =====================================
                for iteration in range(num_splits):

                    reco_theta, reco_s, sinos_masked, _, mask_theta, mask_s = N2I.prepare_batch(
                        sinos.cpu(), iteration, iteration
                    )
                    
                    reco_theta = reco_theta.to(N2I.device)

                    reco_s = reco_s.to(N2I.device)

                    # θ-direction pass
                    _, output_sino_theta, ang, old_ang = N2I.forward(reco_theta, sinos_masked, invariant_inference = True)

                    # s-direction pass
                    _, output_sino_s, ang, old_ang = N2I.forward(reco_s, sinos_masked, invariant_inference = True)


                    
                  # Assuming mask has shape [1, H, W_orig] and we want W_new = 544
                    B, H, W_orig = mask_theta.shape
                    W_new = 544
                    expand_factor = W_new // W_orig  # e.g., 544 / 16 = 34

                    mask_big = torch.ones(1,336,full_size)  # [B, H, W_new]
                    mask_big_s = torch.ones(1,336,full_size)  # [B, H, W_new]

                    for i in range(W_orig):
                        mask_big[:,:,i*expand_factor] = (mask_theta[:,:,i])
                        
                    for i in range(W_orig):
                        mask_big_s[:,:,i*expand_factor] = (mask_s[:,:,i])
                    # average both directions and partitions
                    
                    sino_ds = (output_sino_theta * mask_big.to(output_sino_theta.device) + output_sino_s * mask_big_s.to(output_sino_s.device)) 
                    

                    
                    mask = mask_big.to(sino_ds.device) + mask_big_s.to(sino_ds.device)


                   # print(mask.shape, output_sino.shape, flush = True)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)


                    if final_sino is None:
                        final_sino = torch.zeros_like(sino_ds)

                    final_sino += sino_ds   
                    mask_sum += mask


                final_reco = N2I.fbp_tomosipo(final_sino/mask_sum)
                
                
                mask = create_circular_mask( final_reco.shape[-2],  final_reco.shape[-2], final_reco.shape[-2]//2)
                mask_tensor = torch.from_numpy(mask)
                final_reco = apply_mask(final_reco.detach().cpu(), mask_tensor).to(N2I.device)
                
                final_recos_all_grids.append(final_reco)
                
           # print("Mask sum unique values:", torch.unique(mask_sum), flush=True)
            
            final_recos = torch.stack(final_recos_all_grids, dim=0).mean(dim=0)

            full_recos.append(final_recos)
            Ims.append(ims)
            Recos.append(recos_given)

            err = torch.mean(
                torch.mean((final_recos.squeeze(1) - ims) ** 2, dim=-1),
                dim=-1,
            )
            MSEs.append(err)

    N2I.grid_size = grid_sizes if len(grid_sizes) > 1 else grid_sizes[0]

    full_recos = torch.cat(full_recos, dim=0)
    Ims = torch.cat(Ims, dim=0)
    Recos = torch.cat(Recos, dim=0)
    MSEs = torch.cat(MSEs, dim=0)

    return full_recos, MSEs, Ims, Recos
















#### -----validate average inference stragey for S2I ds------ #######

def validate_average_ds(validation_dataloader, N2I, interpolate=True):
    full_recos = []
    MSEs = []
    Ims = []
    Recos = []
    N2I.random = False
    # normalize grid_size
    grid_sizes = N2I.grid_size
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes]
        
        
    with torch.no_grad():
        for sinos, ims in validation_dataloader:
            sinos = sinos.to(N2I.device)
            ims = ims.to(N2I.device)

            # Baseline reconstruction (full sinogram)
            recos_given = N2I.fbp_tomosipo(sinos)

            batch_size = sinos.shape[0]
            H, W = ims.shape[-2], ims.shape[-1]

            final_recos = torch.zeros(
                (batch_size, 1, H, W),
                device=N2I.device
            )
            final_recos_all_grids = []
            # ---------- LOOP OVER GRID SIZES ----------
            for g in grid_sizes:
                N2I.grid_size = g        # override here

                num_splits = g ** 2
                # =====================================
                # loop over Sparse2Inverse partitions
                # =====================================
                for iteration in range(num_splits):

                    reco_theta, reco_s, _, _, _, _ = N2I.prepare_batch(
                        sinos.cpu(), iteration, iteration
                    )

                    reco_theta = reco_theta.to(N2I.device)
                    reco_s = reco_s.to(N2I.device)

                    # θ-direction pass
                    out_theta, _ = N2I.forward(
                        reco_theta,
                        sinos
                    )

                    # s-direction pass
                    out_s, _ = N2I.forward(
                        reco_s,
                        sinos
                    )

                    # average both directions and partitions
                    final_recos += 0.5 * (out_theta + out_s) / num_splits
                    final_recos_all_grids.append(final_recos)

            # ---------- AVERAGE OVER GRID SIZES ----------
            final_recos = torch.stack(final_recos_all_grids, dim=0).mean(dim=0)

            full_recos.append(final_recos)
            Ims.append(ims)
            Recos.append(recos_given)

            err = torch.mean(
                torch.mean((final_recos.squeeze(1) - ims) ** 2, dim=-1),
                dim=-1,
            )
            MSEs.append(err)

    # restore original grid_size
    N2I.grid_size = grid_sizes if len(grid_sizes) > 1 else grid_sizes[0]

    full_recos = torch.cat(full_recos, dim=0)
    Ims = torch.cat(Ims, dim=0)
    Recos = torch.cat(Recos, dim=0)
    MSEs = torch.cat(MSEs, dim=0)
    N2I.random = False
    return full_recos, MSEs, Ims, Recos




#### -----validate with stragey of Proj2Proj ------ #######
def validate_direct(validation_dataloader, N2I):
    full_recos = []
    MSEs = []
    Ims = []
    Recos = []

    # normalize grid_size
    grid_sizes = N2I.grid_size
    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes]

    with torch.no_grad():
        for sinos, ims in validation_dataloader:
            ims = ims.to(N2I.device)
            recos_given = N2I.fbp_tomosipo(sinos)

            final_recos_all_grids = []

            # ---------- LOOP OVER GRID SIZES ----------
            for g in grid_sizes:
                N2I.grid_size = g

                recos, sinos_out = N2I.prepare_batch_test(sinos)

                input_x_den = recos[:, 0:1].to(N2I.device)

                output_reco, _ = N2I.forward(input_x_den, sinos_out)

                mask = create_circular_mask( output_reco.shape[-2],  output_reco.shape[-2], output_reco.shape[-2]//2)
                 # Convert to torch tensor if needed
                mask_tensor = torch.from_numpy(mask)
                output_reco = apply_mask(output_reco.detach().cpu(), mask_tensor).to(N2I.device)

                final_recos_all_grids.append(output_reco)

            # ---------- AVERAGE OVER GRID SIZES ----------
            final_recos = torch.stack(final_recos_all_grids, dim=0).mean(dim=0)

            full_recos.append(final_recos)
            Ims.append(ims)
            Recos.append(recos_given)

            err = torch.mean(torch.mean((final_recos.squeeze() - ims) ** 2, -1), -1)
            MSEs.append(err)

    # restore original grid_size
    N2I.grid_size = grid_sizes if len(grid_sizes) > 1 else grid_sizes[0]

    full_recos = torch.cat(full_recos, 0)
    Ims = torch.cat(Ims, 0)
    Recos = torch.cat(Recos, 0)
    MSEs = torch.cat(MSEs, 0)

    return full_recos, MSEs, Ims, Recos
