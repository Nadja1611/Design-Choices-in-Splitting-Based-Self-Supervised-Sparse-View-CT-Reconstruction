# -*- coding: utf-8 -*-
"""
Inference script for Sparse2Inverse
Dedicated for noise_intensity=6000
Automatically loads and runs inference for:
- p2p_best_ssim_XXXX.pth
- p2p_best_lpips_XXXX.pth
- s2i_best_ssim_XXXX.pth
- s2i_best_lpips_XXXX.pth
Supports methods:
- P2P
- S2I
- S2I_ds   (double-split)
"""

from operator import gt

import torch, os, re, argparse
import numpy as np
from torch.utils.data import DataLoader

from models import *
from utils import *

import lpips
from piq import haarpsi
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import re
# ------------------------------------------------------------
# ARGPARSE
# ------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-angles","--angles",type=int,default=100)
parser.add_argument("-l","--loss_variant",type=str,default="Sobolev_data")
parser.add_argument("-a","--a",type=float,default=10000.0)
parser.add_argument("-s","--s",type=float,default=1.0)
parser.add_argument("-mode","--mode",type=str,default="mode1")
parser.add_argument("-m","--method",type=str,default="S2I")   # P2P | S2I | S2I_ds
parser.add_argument("-lr","--learning_rate",type=float,default=1e-4)
parser.add_argument("-grid_size","--grid_size",type=int,nargs="+",default=[3])
parser.add_argument("-i","--fill_zeros",action="store_true")
parser.add_argument("-r","--random_mask",action="store_true")
args = parser.parse_args()
# fixed noise intensity for this script

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8

# ------------------------------------------------------------
# FIND LATEST WEIGHTS
# ------------------------------------------------------------

def find_latest(weights_dir, tag):
    # Case 1: exact file, e.g. p2p_best_ssim.pth
    exact_name = f"{tag}.pth"
    exact_path = os.path.join(weights_dir, exact_name)

    if os.path.exists(exact_path):
        return exact_path, None

    # Case 2: epoch file, e.g. p2p_epoch_2400.pth
    pat = re.compile(rf"^{re.escape(tag)}_(\d+)\.pth$")

    best = (-1, None)

    for f in os.listdir(weights_dir):
        m = pat.match(f)
        if m:
            e = int(m.group(1))
            if e > best[0]:
                best = (e, f)

    if best[1] is None:
        print(f"Missing {tag}")
        return None, None

    return os.path.join(weights_dir, best[1]), best[0]

# ------------------------------------------------------------
# EXPERIMENT NAME -- must exactly match the training script
# ------------------------------------------------------------
if "Sobo" in args.loss_variant:
    experiment_name = (
        f"{args.method}_gridsize_{args.grid_size}_loss_"
        f"{args.loss_variant}_a_{args.a}_s_{args.s}"
        f"lr_{args.learning_rate}_angles_{args.angles}_"
        f"random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_{args.mode}"
    )
else:
    experiment_name = (
        f"{args.method}_gridsize_{args.grid_size}_loss_one_grad_step_"
        f"{args.loss_variant}_"
        f"lr_{args.learning_rate}_angles_{args.angles}_"
        f"random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_{args.mode}"
    )

print("Experiment name:", experiment_name)

# ------------------------------------------------------------
# WEIGHT DIRECTORY -- same location and folder name as training
# ------------------------------------------------------------
weights_base = f"../outputs/weights_paper_s2i_{args.mode}"
final_weights_dir = os.path.join(weights_base, experiment_name)

if not os.path.isdir(final_weights_dir):
    available = []
    if os.path.isdir(weights_base):
        available = sorted(
            name for name in os.listdir(weights_base)
            if os.path.isdir(os.path.join(weights_base, name))
        )
    raise FileNotFoundError(
        f"Expected weight directory does not exist:\n{final_weights_dir}\n"
        f"Available experiment folders in {weights_base}:\n" +
        "\n".join(available[:50])
    )

print("Using weight directory:", final_weights_dir)

# ------------------------------------------------------------
# Output root
# ------------------------------------------------------------
if args.mode == "mode1":
    csv_path = f"./predictions_paper_mode1/inferences_summary.csv"
    out_root = os.path.join(f"./predictions_paper_mode1", experiment_name)
else:
    csv_path = f"./predictions_paper_mode2/inferences_summary.csv"
    out_root = os.path.join(f"./predictions_paper_mode2", experiment_name)

print('output ', out_root)
os.makedirs(out_root, exist_ok=True)

# ------------------------------------------------------------
# Model weights inside the selected folder
# ------------------------------------------------------------
print("Final weights directory:", os.listdir(final_weights_dir))
models = {
    "p2p_ssim": find_latest(final_weights_dir, "p2p_best_ssim"),
    "s2i_ssim": find_latest(final_weights_dir, "s2i_best_ssim"),
    "ii_ssim": find_latest(final_weights_dir, "ii_best_ssim"),
    "p2p_psnr": find_latest(final_weights_dir, "p2p_best_psnr"),
    "s2i_psnr": find_latest(final_weights_dir, "s2i_best_psnr"),
    "ii_psnr": find_latest(final_weights_dir, "ii_best_psnr"),
}
open(os.path.join(out_root, "experiment_log.txt"), "a").write(f"{experiment_name} | args={vars(args)} | weights={models} | weights_dir={final_weights_dir}\n")
# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print("Loading test data...")
path_sinos = rf"../all_sinograms_{args.mode}"
sinograms = load_sinograms_to_tensor(path_sinos, nr_angles = args.angles)
sinograms = sinograms.unsqueeze(1)
print(sinograms.shape, flush = True)
sinograms_test = sinograms[950:]

path_reco = rf"../all_reconstructions_{args.mode}"
images = load_reconstructions_to_tensor(path_reco)
images_test = images[950:]
print('NR of images ', images_test.shape, flush = True)

del(images)


# --- 1. Ensure both Tensors are 4D [N, 1, H, W] ---
# (Assuming images and sinograms are loaded as shown in your previous snippet)
images_test = images_test.unsqueeze(1)        # Shape: [N_test, 1, 336, 336]


# --- 2. Per-Image Normalization for Training Set ---
# Find the max value for each reconstruction image across dims 2 and 3
# keepdim=True ensures the shape is [800, 1, 1, 1], allowing flawless broadcasting

# Add a tiny epsilon (1e-8) to prevent any accidental division by zero

# Divide both the sinograms and reconstructions by the reconstruction's max


# --- 3. Per-image normalization, exactly as in training ---
reco_test_maxs = images_test.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
reco_test_maxs = torch.clamp(reco_test_maxs, min=1e-8)

proj_noisy_test = sinograms_test / reco_test_maxs
images_test = images_test / reco_test_maxs




dataset_test = torch.utils.data.TensorDataset(
    proj_noisy_test, images_test.squeeze()
)

print('Normalized test sinograms shape:', proj_noisy_test.shape, flush=True)




Data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)



# ------------------------------------------------------------
# RUN ALL INFERENCES
# ------------------------------------------------------------
def compute_metrics(recos, Ims, device, lpips_fn):
    ssim_list, psnr_list, mse_list, lpips_list, haarpsi_list = [], [], [], [], []
    all_imgs = []

    for i in range(recos.shape[0]):
        reco = recos[i, 0].cpu()
        gt = Ims[i].cpu()
        all_imgs.append(reco.unsqueeze(0))

        reco_np = reco.numpy()

        gt_np = gt.numpy()
        gt_np = np.clip(gt_np, 0.0, 1.0)

      #  factors = np.linspace(0.5,2.5,500)
      #  mse = np.zeros(factors.shape)
      #  for i in range(len(factors)):
      #      factor = factors[i]
     #       reco_scaled = np.clip(reco_np * factor, 0.0, 1.0)
     #       mse[i]= np.mean((gt_np - reco_scaled)**2)

     #   min_i = np.argmin(mse)
      #  print('optimal factor:',factors[min_i])
      ##  reco_np = np.clip(reco_np * factors[min_i], 0.0, 1.0)

        ssim_list.append(structural_similarity(gt_np, reco_np, data_range=gt_np.max()-gt_np.min()))
        psnr_list.append(peak_signal_noise_ratio(gt_np, reco_np))
        mse_list.append(np.mean((gt_np - reco_np)**2))

        reco_lp = (reco.unsqueeze(0).unsqueeze(0)*2 - 1).to(device)
        gt_lp = (gt.unsqueeze(0).unsqueeze(0)*2 - 1).to(device)
        lpips_list.append(lpips_fn(reco_lp, gt_lp).item())

        reco_haar = reco.unsqueeze(0).unsqueeze(0).to(device)
        gt_haar = gt.unsqueeze(0).unsqueeze(0).to(device)
        reco_haar = reco_haar.clamp(0.0, 1.0)
        gt_haar = gt_haar.clamp(0.0, 1.0)
        haarpsi_list.append(haarpsi(reco_haar, gt_haar, data_range=1.0).item())

    return torch.cat(all_imgs, 0), np.array(ssim_list), np.array(psnr_list), np.array(mse_list), np.array(lpips_list), np.array(haarpsi_list)


def compute_metrics_with_factor(recos, Ims, device, lpips_fn, find_constant=False):
    ssim_list, psnr_list, mse_list, lpips_list, haarpsi_list = [], [], [], [], []
    all_imgs = []

    for i in range(recos.shape[0]):
        reco = recos[i, 0].detach().cpu()
        gt = Ims[i].detach().cpu()

        reco_np = reco.numpy().astype(np.float32)
        gt_np = gt.numpy().astype(np.float32)

        # Same as validation
        data_range = gt_np.max() - gt_np.min()

        if data_range <= 0:
            data_range = 1.0

        if find_constant:
            factors = np.linspace(0.3, 2.5, 50)

            best_psnr = -np.inf
            best_mse = None
            best_reco_np = None
            best_factor = None

            for factor in factors:
                reco_scaled_np = factor * reco_np

                psnr_value = peak_signal_noise_ratio(
                    gt_np,
                    reco_scaled_np,
                    data_range=data_range
                )

                mse_value = np.mean((gt_np - reco_scaled_np) ** 2)

                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    best_mse = mse_value
                    best_reco_np = reco_scaled_np
                    best_factor = factor

            reco_np_for_metrics = best_reco_np
            psnr_value = best_psnr
            mse_value = best_mse

        else:
            reco_np_for_metrics = reco_np

            psnr_value = peak_signal_noise_ratio(
                gt_np,
                reco_np_for_metrics,
                data_range=data_range
            )

            mse_value = np.mean((gt_np - reco_np_for_metrics) ** 2)

        ssim_value = structural_similarity(
            gt_np,
            reco_np_for_metrics,
            data_range=data_range
        )

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        mse_list.append(mse_value)

        # Save the reconstruction that was actually evaluated
        reco_eval = torch.from_numpy(reco_np_for_metrics).float()
        all_imgs.append(reco_eval.unsqueeze(0))

        # LPIPS expects approximately [-1, 1]
        reco_lp = (reco_eval.unsqueeze(0).unsqueeze(0) * 2 - 1).to(device)
        gt_lp = (gt.unsqueeze(0).unsqueeze(0) * 2 - 1).to(device)

        lpips_list.append(lpips_fn(reco_lp, gt_lp).item())

        # HaarPSI expects [0, 1] when data_range=1.0
        reco_haar = reco_eval.unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
        gt_haar = gt.unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)

        haarpsi_list.append(
            haarpsi(reco_haar, gt_haar, data_range=1.0).item()
        )

    return (
        torch.cat(all_imgs, 0),
        np.array(ssim_list),
        np.array(psnr_list),
        np.array(mse_list),
        np.array(lpips_list),
        np.array(haarpsi_list)
    )


lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()

for name, (wfile, epoch) in models.items():
    if wfile is None:
        continue

    print(f"\n▶ Running {name} (epoch {epoch})")
    save_dir = os.path.join(out_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # build model
    if args.method == "P2P":
        N2I = Proj2Proj(random=args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    elif args.method == "S2I":
        N2I = Sparse2Inverse_p2p(random=args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    elif args.method == "S2I_ds":
        N2I = Sparse2Inverse_ds_all_combinations(random = args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    N2I.net_denoising.load_state_dict(torch.load(wfile, map_location=device))
    N2I.net_denoising.eval()

    # inference types direct vs average vs P-invariant
    with torch.no_grad():
        if name.startswith("p2p"):
            print(name)
            full_recos, _, Ims, _ = validate_direct(Data_loader_test, N2I)
        elif name.startswith("s2i"):
            print(name)
            if args.method == "S2I_ds":
                full_recos, _, Ims, _ = validate_average_ds(Data_loader_test, N2I)
            else:
                full_recos, _, Ims, _ = validate_average(Data_loader_test, N2I)
        elif name.startswith("ii"):
            print(name)
            if args.method == "S2I_ds":
                full_recos, _, Ims, _ = validate_P_invariant_doublesplit(Data_loader_test, N2I)
            else:
                full_recos, _, Ims, _ = validate_P_invariant(Data_loader_test, N2I)


    all_imgs, ssim, psnr, mse, lpips_vals, haarpsi_vals = compute_metrics_with_factor(full_recos, Ims, device, lpips_fn)

    torch.save(all_imgs, os.path.join(save_dir, "all_reconstructions.pt"))
    np.savez(os.path.join(save_dir, "metrics.npz"), ssim=ssim, psnr=psnr, mse=mse, lpips=lpips_vals, haarpsi=haarpsi_vals)

    print(f"{name} finished. Mean SSIM: {np.mean(ssim):.4f}")

    # write summary CSV
    summary_csv = csv_path
    df_row = pd.DataFrame([{
        "method_name": name,
        "experiment_name": experiment_name,
        "grid_size": "_".join(map(str, args.grid_size)),
        "angles": args.angles,
        "ssim": float(np.mean(ssim)),
        "psnr": float(np.mean(psnr)),
        "lpips": float(np.mean(lpips_vals)),
        "haarpsi": float(np.mean(haarpsi_vals))
    }])
    if os.path.exists(summary_csv):
        df_row.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_csv, mode="w", header=True, index=False)

    print(f" Metrics appended to {summary_csv}")