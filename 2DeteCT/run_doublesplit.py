import torch
import cv2 as cv
import os
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
from utils import * 
from torch.utils.tensorboard import SummaryWriter
import lpips



# %%
parser = argparse.ArgumentParser(
    description="Arguments for segmentation network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used? Options are MSE_data, MSE_image, Sobolev_data",
    default="Sobolev_data",
)

parser.add_argument(
    "-a",
    "--a",
    type=float,
    help="parameter a in sobolev norm",
    default=10000.0,
)
parser.add_argument(
    "-s",
    "--s",
    type=float,
    help="parameter s in sobolev norm",
    default=1.,
)
parser.add_argument(
    "-angles",
    "--angles",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=16,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-4,
)
parser.add_argument(
    "-o",
    "--logdir",
    type=str,
    help="directory for log files",
    default="/home/nadja/Noisier2Inverse_Github/Sparse2Inverse/logs",
)
parser.add_argument(
    "-s_interpolation",
    "--s_interpolation",
    type=str,
    help="should the s interpolation be applied",
    default="no",
)
parser.add_argument(
    "-grid_size",
    "--grid_size",
    type=int,
    nargs="+",          # allow one or more ints
    help="grid size (either one number or multiple for random choice)",
    default=[3]
)

parser.add_argument(
    "-r", "--random_mask",
    action="store_true",
    help="enable random masking"
)
parser.add_argument(
    "-interpolate",
    "--interpolate",
    action="store_true",
    help="interpolation in angular directoin",
    default = True)
parser.add_argument(
    "-method",
    "--method",
    type = str,
    help="choose splitting that should be used, S2I , P2P, S2I_ds are the options",
    default = "S2I_ds")
parser.add_argument(
    "-device",
    "--device",
    type = str,
    help="choose the device which is used for training",
    default = "cuda:0")
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type = int,
    help="batch size used for training",
    default = 32)
parser.add_argument(
    "-show_images",
    "--show_images",
    type = bool,
    help = "1 to show images and 0 to save them",
    default = False)
parser.add_argument(
    "-mode",
    "--mode",
    type = str,
    help="which mode of 2detect data should be used, mode1 or mode2",
    default = "mode1")

parser.add_argument(
    "-i", "--fill_zeros",
    action="store_true",
    help="enable interpolation in angular direction"
)



'>>-------------------------------------------------------------------------<<'
' Parse arguments and set random seed'
'>>-------------------------------------------------------------------------<<'

torch.manual_seed(0)

args = parser.parse_args()

number_angles = args.angles
device = args.device
loss_variant = args.loss_variant
batch_size = args.batch_size
mode = args.mode
print('random and fill', args.random_mask, args.fill_zeros, flush = True)

'>>-------------------------------------------------------------------------<<'
' Loading and augmenting image data. Computing sinograms'
'>>-------------------------------------------------------------------------<<'
path_sinos = rf"../all_sinograms_{mode}"
sinograms = load_sinograms_to_tensor(path_sinos, nr_angles = args.angles)
sinograms = sinograms.unsqueeze(1)
print(sinograms.shape, flush = True)
sinograms_test = sinograms[950:]
sinograms = sinograms[:100]

path_reco = rf"../all_reconstructions_{mode}"
images = load_reconstructions_to_tensor(path_reco)
images_training = images[:100]
images_test = images[950:]
print('NR of images ', images_training.shape, images_test.shape, flush = True)

del(images)

'>>-------------------------------------------------------------------------<<'
' Adding noise to the projection data'
'>>-------------------------------------------------------------------------<<'


proj_noisy = sinograms
proj_noisy_test = sinograms_test

'>>-------------------------------------------------------------------------<<'
' Generating dataset'
'>>-------------------------------------------------------------------------<<'
    # --- 1. Ensure both Tensors are 4D [N, 1, H, W] ---
    # (Assuming images and sinograms are loaded as shown in your previous snippet)
images_training = images_training.unsqueeze(1)    # Shape: [800, 1, 336, 336]
images_test = images_test.unsqueeze(1)        # Shape: [N_test, 1, 336, 336]


# --- 2. Per-Image Normalization for Training Set ---
# Find the max value for each reconstruction image across dims 2 and 3
# keepdim=True ensures the shape is [800, 1, 1, 1], allowing flawless broadcasting
reco_train_maxs = images_training.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

# Add a tiny epsilon (1e-8) to prevent any accidental division by zero
reco_train_maxs = torch.clamp(reco_train_maxs, min=1e-8)

# Divide both the sinograms and reconstructions by the reconstruction's max
proj_noisy = sinograms / reco_train_maxs
images_training = images_training / reco_train_maxs


# --- 3. Per-Image Normalization for Test Set ---
reco_test_maxs = images_test.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
reco_test_maxs = torch.clamp(reco_test_maxs, min=1e-8)

proj_noisy_test = sinograms_test / reco_test_maxs
images_test = images_test / reco_test_maxs


# --- 4. Create Datasets ---
dataset = torch.utils.data.TensorDataset(
    proj_noisy, images_training.squeeze()
)

dataset_test = torch.utils.data.TensorDataset(
    proj_noisy_test, images_test.squeeze()
)

print('Normalized training images shape:', images_training.shape, flush=True)
print('Normalized training sinograms shape:', proj_noisy.shape, flush=True)




Data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
Data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)



'>>-------------------------------------------------------------------------<<'
'Define training details and prepare output folders / tensorboard'
'>>-------------------------------------------------------------------------<<'

N_epochs = 10000
learning_rate = args.learning_rate



if 'Sobo' in args.loss_variant:
    experiment_name = (
    f"{args.method}_gridsize_{args.grid_size}_loss_"
    f"{args.loss_variant}_a_{args.a}_s_{args.s}"
    f"lr_{args.learning_rate}_angles_{args.angles}_random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_{mode}"
)
else:
    experiment_name = (
    f"{args.method}_gridsize_{args.grid_size}_loss_one_grad_step_"
    f"{args.loss_variant}_"
    f"lr_{args.learning_rate}_angles_{args.angles}_random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_{mode}"
)


# Define TensorBoard log path
newpath = os.path.join("../weights/", experiment_name)
weights_dir = os.path.join(f"../outputs/weights_paper_s2i_{mode}", experiment_name)
# Define TensorBoard log path
writer = SummaryWriter(log_dir=os.path.join(f"../tensorboards/tensorboard_ii_{args.angles}_random_{args.random_mask}", experiment_name))

if not os.path.exists(newpath):
    os.makedirs(newpath)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

print('Training with loss variant: ', loss_variant)

'>>-------------------------------------------------------------------------<<'
' Define model and optimizer'
'>>-------------------------------------------------------------------------<<'

N2I = Sparse2Inverse_ds_all_combinations(random = args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
N2I_optimizer = optim.Adam(N2I.net_denoising.parameters(), lr=learning_rate)


lpips_fn = lpips.LPIPS(net='alex').to(N2I.device)
lpips_fn.eval()
# Initialize lists for tracking performance

lpips_fn = lpips.LPIPS(net='alex').to(N2I.device)
lpips_fn.eval()
# Initialize lists for tracking performance
l2_list, all_MSEs, all_ssim, all_psnr, all_mse =[], [], [], [], []

all_ssim_p2p, all_psnr_p2p, all_mse_p2p = [], [], []
all_ssim_s2i, all_psnr_s2i, all_mse_s2i = [], [], []
all_ssim_ii, all_psnr_ii, all_mse_ii = [], [], []

all_lpips_p2p, all_lpips_s2i, all_lpips_ii = [], [], []
best_lpips_p2p, best_lpips_s2i, best_lpips_ii = float("inf"), float("inf"), float("inf")

max_ssim_p2p, max_psnr_p2p = 0, 0
max_ssim_s2i, max_psnr_s2i = 0, 0
max_ssim_ii, max_psnr_ii = 0, 0

old_ssim_p2p, old_psnr_p2p = 0, 0
old_ssim_s2i, old_psnr_s2i = 0, 0
old_ssim_ii, old_psnr_ii = 0, 0


# Enable mixed precision training 
scaler = torch.cuda.amp.GradScaler()

old_psnr = 0.1
old_ssim = 0.1



G2 = args.grid_size[0] ** 2

'>>-------------------------------------------------------------------------<<'
' Training loop'
'>>-------------------------------------------------------------------------<<'
########################### Now training starts ##############
for epoch in range(N_epochs):
    running_loss = 0
    running_L2_loss = 0


    for sinos, ims in Data_loader:
        iteration_theta = torch.randint(0, G2, (1,)).item()
        iteration_s     = torch.randint(0, G2, (1,)).item()
        N2I_optimizer.zero_grad()

        # prepare batch

        reco_theta, reco_s, target, sinos, mask_theta, mask_s = \
            N2I.prepare_batch(sinos, iteration_theta, iteration_s)

        sinos = sinos.to(N2I.device, non_blocking=True)
        target = target.to(N2I.device, non_blocking=True)

        B = sinos.shape[0]

        # =============================
        # θ-direction pass
        # =============================
        input_x_den = reco_theta.to(N2I.device)

        output_reco_theta, output_sino_theta = N2I.forward(
            input_x_den, sinos
        )

        #mask_theta = mask_theta.unsqueeze(1).repeat(B, 1, 1, 1).to(N2I.device)
        mask_theta = mask_theta.unsqueeze(1).expand(B, -1, -1, -1).to(N2I.device)

        if args.loss_variant == 'Sobolev_data':
            loss_theta = sobolev_norm_fourier(
                output_sino_theta * mask_theta,
                sinos * mask_theta,
                s=args.s, a=args.a
            )
        else:
            loss_theta = torch.nn.functional.mse_loss(
                output_sino_theta * mask_theta,
                sinos * mask_theta
            )

        # =============================
        # s-direction pass
        # =============================
        input_x_den = reco_s.to(N2I.device)

        output_reco_s, output_sino_s = N2I.forward(
            input_x_den, sinos
        )

#        mask_s = mask_s.unsqueeze(1).repeat(B, 1, 1, 1).to(N2I.device)
        mask_s  = mask_s.unsqueeze(1).expand(B, -1, -1, -1).to(N2I.device)

        if args.loss_variant == 'Sobolev_data':
            loss_s = sobolev_norm_fourier(
                output_sino_s * mask_s,
                sinos * mask_s,
                s=args.s, a=args.a
            )
        else:
            loss_s = torch.nn.functional.mse_loss(
                output_sino_s * mask_s,
                sinos * mask_s
            )
            
        ### combine the outputs ####
        output_reco = (output_reco_s + output_reco_theta)/2
        # =============================
        # total loss + backward
        # =============================
        loss = loss_theta + loss_s


        with torch.no_grad():
            l2_loss = torch.nn.functional.mse_loss(
                output_reco.float(), target.float().to(device)
            )

        scaler.scale(loss).backward()
        scaler.step(N2I_optimizer)
        scaler.update()
        #loss.backward()
      #  N2I_optimizer.step()
        running_loss += loss.item()
        running_L2_loss += l2_loss.item()
        del (
            output_reco_theta, output_sino_theta,
            output_reco_s, output_sino_s,
            loss_theta, loss_s, loss
        )

    l2_list.append(running_loss)
    
                

            

    '**---------------------------------------------------------------------**'
    ' Validation of the model'
    '**---------------------------------------------------------------------**'
   
    if epoch % 5 == 0:

        full_recos_p2p, MSEs_p2p, Ims_p2p, Recos_p2p = validate_direct(Data_loader_test, N2I)
        full_recos_s2i, MSEs_s2i, Ims_s2i, Recos_s2i = validate_average_ds(Data_loader_test, N2I)
        full_recos_ii, MSEs_ii, Ims_ii, Recos_ii = validate_P_invariant_doublesplit(Data_loader_test, N2I)


    '**---------------------------------------------------------------------**'
    ' Compute validation metrics and solve the model weights'
    '**---------------------------------------------------------------------**'
    
    if epoch % 10 == 0:

        mean_ssim_p2p, mean_psnr_p2p, mean_mse_p2p = compute_validation_metrics(full_recos_p2p, Ims_p2p)
        mean_ssim_s2i, mean_psnr_s2i, mean_mse_s2i = compute_validation_metrics(full_recos_s2i, Ims_s2i)
        mean_ssim_ii, mean_psnr_ii, mean_mse_ii = compute_validation_metrics(full_recos_ii, Ims_ii)


        all_ssim_p2p.append(mean_ssim_p2p)
        all_psnr_p2p.append(mean_psnr_p2p)
        all_mse_p2p.append(mean_mse_p2p)

        all_ssim_s2i.append(mean_ssim_s2i)
        all_psnr_s2i.append(mean_psnr_s2i)
        all_mse_s2i.append(mean_mse_s2i)

        all_ssim_ii.append(mean_ssim_ii)
        all_psnr_ii.append(mean_psnr_ii)
        all_mse_ii.append(mean_mse_ii)

        max_ssim_p2p = max(max_ssim_p2p, mean_ssim_p2p)
        max_psnr_p2p = max(max_psnr_p2p, mean_psnr_p2p)

        max_ssim_s2i = max(max_ssim_s2i, mean_ssim_s2i)
        max_psnr_s2i = max(max_psnr_s2i, mean_psnr_s2i)

        max_ssim_ii = max(max_ssim_ii, mean_ssim_ii)
        max_psnr_ii = max(max_psnr_ii, mean_psnr_ii)
        


        # -------- LPIPS (lower is better) --------
        # full_recos_* is typically [B, K, H, W]; we compare channel 0 recon vs GT
        lpips_p2p = compute_lpips(lpips_fn, full_recos_p2p[:, 0:1], Ims_p2p, N2I.device)
        lpips_s2i = compute_lpips(lpips_fn, full_recos_s2i[:, 0:1], Ims_s2i, N2I.device)
        lpips_ii = compute_lpips(lpips_fn, full_recos_ii[:, 0:1], Ims_ii, N2I.device)


        all_lpips_p2p.append(lpips_p2p)
        all_lpips_s2i.append(lpips_s2i)
        all_lpips_ii.append(lpips_ii)

       # =========================
        # SAVE MODELS for p2p (direct inference), s2i (average of two splits), and ii (invariant inference)
        # =========================

        if epoch % 100 == 0:
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, f"p2p_epoch_{epoch}.pth"))

        elif epoch > 100 and mean_ssim_p2p > old_ssim_p2p:
            old_ssim_p2p = mean_ssim_p2p
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, "p2p_best_ssim.pth"))

        if epoch % 100 == 0:
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, f"s2i_epoch_{epoch}.pth"))

        elif epoch > 100 and mean_ssim_s2i > old_ssim_s2i:
            old_ssim_s2i = mean_ssim_s2i
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, "s2i_best_ssim.pth"))

        if epoch % 100 == 0:
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, f"ii_epoch_{epoch}.pth"))

        elif epoch > 100 and mean_ssim_ii > old_ssim_ii:
            old_ssim_ii = mean_ssim_ii
            torch.save(N2I.net_denoising.state_dict(),
                    os.path.join(weights_dir, "ii_best_ssim.pth"))

                    
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))

        for row, (Ims, full_recos, Recos, name) in enumerate([
            (Ims_p2p, full_recos_p2p, Recos_p2p, "P2P"),
            (Ims_s2i, full_recos_s2i, Recos_s2i, "S2I"),
            (Ims_ii, full_recos_ii, Recos_ii, "II")

        ]):

            for ax, img, title in zip(
                axes[row],
                [Ims[0], full_recos[0, 0], Recos[0, 0]],
                ["Ground Truth", "Final Reconstruction", "Input"]
            ):
                im = ax.imshow(img.detach().cpu(), cmap="gray", vmin=0, vmax = 1)
                ax.set_title(f"{name} - {title}")
                ax.axis("off")

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        del full_recos_p2p, MSEs_p2p, Ims_p2p, Recos_p2p
        del full_recos_s2i, MSEs_s2i, Ims_s2i, Recos_s2i
        del full_recos_ii, MSEs_ii, Ims_ii, Recos_ii
        torch.cuda.empty_cache()
        gc.collect()
        # TensorBoard: both inferences in ONE figure
        writer.add_figure("Reconstruction_Comparison/P2P_vs_S2I", fig, global_step=epoch)
        plt.close(fig)

        # -------- TensorBoard logging --------
        writer.add_scalar("P2P/SSIM_Last", mean_ssim_p2p, epoch)
        writer.add_scalar("P2P/PSNR_Last", mean_psnr_p2p, epoch)
        writer.add_scalar("P2P/MSE_Last",  mean_mse_p2p,  epoch)

        writer.add_scalar("S2I/SSIM_Last", mean_ssim_s2i, epoch)
        writer.add_scalar("S2I/PSNR_Last", mean_psnr_s2i, epoch)
        writer.add_scalar("S2I/MSE_Last",  mean_mse_s2i,  epoch)

        writer.add_scalar("II/SSIM_Last", mean_ssim_ii, epoch)
        writer.add_scalar("II/PSNR_Last", mean_psnr_ii, epoch)
        writer.add_scalar("II/MSE_Last",  mean_mse_ii,  epoch)

        writer.add_scalar("P2P/SSIM_Max", max_ssim_p2p, epoch)
        writer.add_scalar("S2I/SSIM_Max", max_ssim_s2i, epoch)
        writer.add_scalar("II/SSIM_Max", max_ssim_ii, epoch)


        writer.add_scalar("P2P/PSNR_Max", max_psnr_p2p, epoch)
        writer.add_scalar("S2I/PSNR_Max", max_psnr_s2i, epoch)
        writer.add_scalar("II/PSNR_Max", max_psnr_ii, epoch)


        writer.add_scalar("P2P/LPIPS_Last", lpips_p2p, epoch)
        writer.add_scalar("S2I/LPIPS_Last", lpips_s2i, epoch)
        writer.add_scalar("II/LPIPS_Last", lpips_ii, epoch)


        writer.add_scalar("Training/Loss", l2_list[-1], epoch)

        print(f"\nEpoch {epoch}", flush = True)
        print(f"P2P → SSIM {mean_ssim_p2p:.4f} | PSNR {mean_psnr_p2p:.2f}", flush = True)
        print(f"S2I → SSIM {mean_ssim_s2i:.4f} | PSNR {mean_psnr_s2i:.2f}", flush = True)
        print(f"II → SSIM {mean_ssim_ii:.4f} | PSNR {mean_psnr_ii:.2f}", flush = True)

writer.close()        