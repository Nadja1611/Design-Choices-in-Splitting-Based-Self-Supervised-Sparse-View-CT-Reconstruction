# BenchS2I

**BenchS2I** is a benchmark and research framework for evaluating and extending
**splitting-based self-supervised learning methods for sparse-view CT reconstruction**.

The repository provides implementations of:

- **S2I** — angular splitting
- **P2P** — lattice splitting
- **S2I_ds** — joint angular and detector splitting (DoubleSplit)

The code is organized into separate experiment pipelines for the **LoDoPaB-CT**
and **2DeteCT** datasets.

---

## Dataset Overview

### LoDoPaB-CT

The experiments in [`lodopab/`](lodopab/) are based on the LoDoPaB-CT dataset.

- Original dataset: https://zenodo.org/records/3384092
- Preprocessed data used to reproduce the experiments:
  https://drive.google.com/drive/folders/1CyKf-GlcJ0v8jNaRDHmzvdmTyrnI6QBo?usp=drive_link

The LoDoPaB utilities load and prepare CT images stored as PyTorch `.pt` files
and generate sparse-view noisy sinograms during training and evaluation.

### 2DeteCT

The experiments in [`2DeteCT/`](2DeteCT/) are based on the 2DeteCT dataset.

Preprocessed mode-1 data used by the repository:

- Sinograms:
  https://drive.google.com/drive/folders/1YWk3CH6CSc4kEN20zCr2dh462vTQYtDF?usp=sharing
- Reconstructions:
  https://drive.google.com/drive/folders/1fsn2fK0liVrvm2kfSeHn8hbwaRs6YV81?usp=drive_link

The preprocessing scripts in `2DeteCT/` convert and reorganize the original
data into the directory layout expected by the training and inference scripts.

---

## Repository Structure

```text
Design-Choices-in-Splitting-Based-Self-Supervised-Sparse-View-CT-Reconstruction/
├── README.md
├── bash.sh
│
├── lodopab/
│   ├── run.py
│   │   └── Training entry point for S2I and P2P on LoDoPaB-CT
│   ├── run_doublesplit.py
│   │   └── Training entry point for DoubleSplit / S2I_ds
│   ├── models.py
│   │   └── Reconstruction networks and splitting-based model classes
│   ├── utils.py
│   │   └── CT operators, losses, validation, metrics, and shared utilities
│   └── utils_lodopab.py
│       └── LoDoPaB-specific data loading and preprocessing utilities
│
└── 2DeteCT/
    ├── run.py
    │   └── Training entry point for S2I and P2P on 2DeteCT
    ├── run_doublesplit.py
    │   └── Training entry point for DoubleSplit / S2I_ds
    ├── inference.py
    │   └── Checkpoint loading, reconstruction, and quantitative evaluation
    ├── models.py
    │   └── Reconstruction networks and splitting-based model classes
    ├── utils.py
    │   └── CT operators, losses, validation, metrics, and shared utilities
    ├── preprocess_2detect.py
    │   └── Preprocessing of the original 2DeteCT data
    ├── restructure_2detect.py
    │   └── Reorganization of preprocessed data into the expected layout
    └── run commands
        └── Example commands for running 2DeteCT experiments
```

The two dataset directories are largely self-contained. Run scripts from the
corresponding directory so that local imports such as `models` and `utils`
resolve to the correct implementation.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nadja1611/Design-Choices-in-Splitting-Based-Self-Supervised-Sparse-View-CT-Reconstruction.git
cd Design-Choices-in-Splitting-Based-Self-Supervised-Sparse-View-CT-Reconstruction
```

### 2. Install dependencies

Install the PyTorch version appropriate for your CUDA environment first, then
install the remaining dependencies required by the selected dataset pipeline.

The code uses packages including PyTorch, NumPy, SciPy, scikit-image,
Matplotlib, LPIPS, PIQ, Tomosipo, and LION.

### 3. Select the experiment pipeline

For LoDoPaB-CT experiments:

```bash
cd lodopab
```

For 2DeteCT experiments:

```bash
cd 2DeteCT
```

Place the downloaded or preprocessed data at the paths expected by the scripts,
or update the path variables in the corresponding training and inference files.

---

## Training

### Common arguments

| Argument | Description | Default |
|---|---|---:|
| `--method` | Splitting method: `S2I`, `P2P`, or `S2I_ds` | `S2I` |
| `--angles` | Number of projection angles | `16` |
| `--grid_size` | Detector/lattice splitting grid size | `[3]` |
| `--number_training_imgs` | Number of training images, where supported | `1000` |
| `--fill_zeros`, `-i` | Enable interpolation in the angular direction | disabled |
| `--random_mask`, `-r` | Use random masking | disabled |
| `--correlated_noise` | Enable correlated noise, where supported | disabled |
| `--loss_variant`, `-l` | `MSE_data`, `MSE_image`, or `Sobolev_data` | `Sobolev_data` |
| `--a`, `--s` | Sobolev-norm parameters | `10000`, `1` |
| `--learning_rate`, `-lr` | Learning rate | `1e-4` |
| `--batch_size` | Training batch size | `32` |
| `--device` | Training device | `cuda:0` |

Exact options can differ slightly between the LoDoPaB and 2DeteCT scripts.
Run `python run.py --help` or inspect the argument definitions in the relevant
entry point before launching an experiment.

### S2I: angular splitting

Run from either `lodopab/` or `2DeteCT/`:

```bash
python run.py \
    --method S2I \
    --angles 16 \
    --fill_zeros \
    --loss_variant MSE_data
```

### P2P: lattice splitting

```bash
python run.py \
    --method P2P \
    --angles 16 \
    --grid_size 2 \
    --fill_zeros \
    --loss_variant MSE_data
```

### DoubleSplit / S2I_ds

```bash
python run_doublesplit.py \
    --method S2I_ds \
    --loss_variant MSE_data \
    --grid_size 2 \
    --angles 16 \
    --fill_zeros
```

To enable correlated noise in scripts that support it, add:

```bash
--correlated_noise
```

---

## Inference

The tracked inference entry point is located in:

```text
2DeteCT/inference.py
```

Run it from the `2DeteCT/` directory using the same configuration values used
during training. These values are required to reconstruct the experiment name
and locate the corresponding checkpoint directory.

Example:

```bash
cd 2DeteCT

python inference.py \
    --method S2I \
    --grid_size 3 \
    --angles 16 \
    --loss_variant MSE_data \
    --learning_rate 0.0001 \
    --fill_zeros
```

Checkpoint and output paths are constructed by the scripts from the experiment
arguments. Therefore, the method, angle count, grid size, learning rate, loss,
masking, interpolation, noise settings, and dataset mode must match the
training run.

---

## Citation

```bibtex
@article{benchs2i,
  title   = {Benchmark and Extensions of Splitting-based Methods for
             Self-Supervised Sparse-View CT},
  author  = {Nadja et al.},
  journal = {...},
  year    = {2025}
}
```
