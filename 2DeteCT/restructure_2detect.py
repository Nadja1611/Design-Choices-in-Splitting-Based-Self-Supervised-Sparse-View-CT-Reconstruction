import os
import shutil
import re

# Define the base directory and the target output directories
base_dir = r"/scratch/nadja/BenchS2I_2detect/BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT/LION/LION/data_loaders/2deteCT_processed"
output_sinograms = os.path.join(base_dir, "all_sinograms_mode1")
output_reconstructions = os.path.join(base_dir, "all_reconstructions_mode1")

# Create the new folders if they don't exist yet
os.makedirs(output_sinograms, exist_ok=True)
os.makedirs(output_reconstructions, exist_ok=True)

print("Starting to sort and rename files...")

# Iterate through everything inside the processed folder
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    
    # Skip our new output directories and ensure we are looking at slice folders
    if item in ["all_sinograms", "all_reconstructions"] or not os.path.isdir(item_path):
        continue
    
    # Extract the slice number from the folder name
    slice_match = re.search(r'\d+', item)
    if slice_match:
        slice_num = slice_match.group()
    else:
        # Fallback to the whole folder name if no number is found
        slice_num = item 

    # --- Process Mode 1 (Sinogram, Dark, Flat) ---
    mode1_dir = os.path.join(item_path, "mode1")
    
    # Dictionary tracking the filename we want to find and its corresponding new prefix
    mode1_files = {
        "sinogram.npy": f"sinogram_slice_{slice_num}.npy",
        "dark.npy": f"dark_slice_{slice_num}.npy",
        "flat.npy": f"flat_slice_{slice_num}.npy"
    }
    
    for original_name, new_name in mode1_files.items():
        source_path = os.path.join(mode1_dir, original_name)
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(output_sinograms, new_name))
        else:
            print(f"Warning: {original_name} missing in {item}/mode1")

    # --- Process Mode 3 (Reconstruction) ---
    recon_source = os.path.join(item_path, "mode3", "reconstruction.npy")
    if os.path.exists(recon_source):
        new_recon_name = f"reconstruction_slice_{slice_num}.npy"
        shutil.copy2(recon_source, os.path.join(output_reconstructions, new_recon_name))
    else:
        print(f"Warning: reconstruction.npy missing in {item}/mode3")

print("\nTask complete! Your folders are ready:")
print(f"-> Sinograms, Darks & Flats: {output_sinograms}")
print(f"-> Reconstructions: {output_reconstructions}")