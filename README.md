# BenchS2I

**BenchS2I** is a benchmark and research framework for evaluating and extending **splitting-based self-supervised learning methods for sparse-view CT reconstruction**. The repository provides implementations of:

- **S2I** – angular splitting  
- **P2P** – lattice splitting  
- **S2I_ds** – joint angular + detector splitting (DoubleSplit)  

It includes standardized experimental setups based on the **LoDoPaB-CT dataset**, with support for various noise models, interpolation, and flexible loss functions.

---

##  Dataset Overview

The experiments are based on the **LoDoPaB-CT Dataset**.

Download the dataset here:  
 https://zenodo.org/records/3384092

###  Data Source

The dataset is derived from clinical CT scans, converted to `.pt` files for efficient training and evaluation. The repository includes scripts to process the LoDoPaB-CT testing split into training and test sets.

---

##  Repository Structure

```text
BenchS2I/
├── run.py                  # Training for S2I and P2P
├── run_doublesplit.py      # Training for joint angular + detector splitting
├── inference.py            # Inference and evaluation
├── models.py               # Model architectures
├── utils.py                # General utilities
├── utils_lodopab.py        # LoDoPaB-specific utilities
├── requirements.txt
└── README.md
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nadja1611/BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT.git
cd BenchS2I-Benchmark-and-Extensions-of-Splitting-based-methods-for-Self-Supervised-Sparse-View-CT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and prepare the dataset

Download the LoDoPaB-CT dataset from: https://zenodo.org/records/3384092  
Then convert `.hdf5` files to `.pt` files if needed (see `utils_lodopab.py`).

---

##  Training

### Common arguments

| Argument | Description | Default |
|----------|------------|---------|
| `--method` | Splitting method: `S2I`, `P2P`, or `S2I_ds` | `S2I` |
| `--angles` | Number of projection angles | 16 |
| `--grid_size` | Detector/lattice splitting grid size (list or single int) | [3] |
| `--number_training_imgs` | Number of training images | 1000 |
| `--fill_zeros` / `-i` | Enable angular interpolation | False |
| `--random_mask` | Use random masking | False |
| `--correlated_noise` | Apply correlated Poisson noise | False |
| `--loss_variant` | Loss function (`MSE_data`, `MSE_image`, `Sobolev_data`) | Sobolev_data |
| `--a`, `--s` | Sobolev norm parameters | a=10000, s=1 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--batch_size` | Batch size | 32 |
| `--device` | Device for training (`cuda:0` or `cpu`) | cuda:0 |

---

### 1️ Angular Splitting (S2I)

Example:

```bash
python run.py \
    --method S2I \
    --angles 16 \
    --number_training_imgs 1000 \
    --fill_zeros \
    --loss_variant MSE_data
```

---

### 2️ Latttice Splitting (P2P)

Example:

```bash
python run.py \
    --method P2P \
    --angles 16 \
    --grid_size 2 \
    --number_training_imgs 1000 \
    --fill_zeros \
    --loss_variant MSE_data
```

---

### 3️ Joint Angular + Detector Splitting (DoubleSplit / S2I_ds)

Example:

```bash
python run_doublesplit.py \
    -l MSE_data \
    -grid_size 2 \
    --angles 16 \
    --number_training_imgs 1000 \
    -i \
```

> Notes:  
> - `-i` / `--fill_zeros` works for **all methods**, not only correlated noise.  
> - To enable correlated noise, add `--correlated_noise`.  
> - Random masking is enabled via `--random_mask`.

---

##  Example: Correlated Noise with Interpolation

```bash
python run_doublesplit.py \
    -l MSE_data \
    -grid_size 2 \
    --angles 16 \
    --number_training_imgs 1000 \
    -i \
    --correlated_noise \
    --method S2I_ds
```

---

##  Inference

After training, run inference using:
```bash

methods=( "S2I" "P2P")
grid_sizes=(2 3 4)
angles_list=(16 32)
noise_intensity=(3500)

# Logging
mkdir -p logs

for method in "${methods[@]}"; do
  for grid in "${grid_sizes[@]}"; do
    for ang in "${angles_list[@]}"; do
      for ni in "${noise_intensity[@]}"; do

        echo "Running: method=$method grid=$grid angles=$ang noise=$ni"

        python inference.py \
            --method "$method" \
            --grid_size "$grid" \
            --angles "$ang" \
            --noise_intensity "$ni" \
            --random_mask \
        | tee "logs/${method}_g${grid}_a${ang}_ni${ni}.log"

      done
    done
  done
done
```
---

## 📖 Citation

```bibtex
@article{benchs2i,
  title={Benchmark and Extensions of Splitting-based Methods for Self-Supervised Sparse-View CT},
  author={Nadja et al.},
  journal={...},
  year={2025}
}
```
