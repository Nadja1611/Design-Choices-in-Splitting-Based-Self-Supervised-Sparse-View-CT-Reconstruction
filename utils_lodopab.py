import os
import torch
import numpy as np

def get_images_from_pt(path, amount_of_images='all', scale_number=1):
    all_images = []
    all_pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]
    all_pt_files.sort()  # sort alphabetically by filename

    if amount_of_images == 'all':
        selected_files = all_pt_files
        print(selected_files, flush = True)
    else:
        # random sample of pt files
        idxs = np.random.permutation(len(all_pt_files))[:amount_of_images]
        selected_files = [all_pt_files[i] for i in idxs]

    for fname in selected_files:
        data = torch.load(os.path.join(path, fname), map_location="cpu")
        # Your saved gt is shape (N,1,H,W)
        if isinstance(data, dict):
            gt = data["ground_truth"]  # if you saved dicts
        else:
            gt = data  # if you saved pure tensor
        gt = gt.squeeze(1).numpy()  # (N,H,W)

        for img in gt:

            all_images.append(img)

    return np.array(all_images, dtype='float32')
