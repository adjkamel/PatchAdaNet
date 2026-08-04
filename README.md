# PatchAdaNet

**PatchAdaNet: Unsupervised Class-Incremental Learning using Local Patches and Global Adapters**

PatchAdaNet is a replay-free, unsupervised class-incremental learning framework that separates universal structural feature learning from task-specific adaptation. It operates in two stages:

- **Stage 1 — Patch Autoencoder Pretraining:** Each input image is divided into a grid of non-overlapping patches. An independent autoencoder (CNN-PatchAE or ViT-PatchAE) is trained per patch position on external data to capture low- to mid-level structural features (edges, textures, shapes) that generalize across tasks. After pretraining, all patch encoder weights are permanently frozen.

- **Stage 2 — Continual Global Model Training:** The frozen patch-level representations are aggregated and passed through a global autoencoder (CNN-ImageAE or ViT-ImageAE) trained sequentially across tasks. Learnable adapter layers refine the shared frozen representations into task-specific global image representations. Two complementary regularizers prevent forgetting: the **Global Feature Centroid (GFC)** penalizes representational drift in activation space, and **EWC** constrains parameter drift in weight space — both applied exclusively to the adapter layers.

After each task, a linear classifier is trained on the frozen bottleneck features concatenated with the deepest GFC centroid for class-incremental or task-incremental evaluation.

---

## Environment

```
Python  >= 3.8
PyTorch >= 1.12
CUDA    >= 11.0  (recommended)
```

Install dependencies:

```bash
pip install torch torchvision scikit-learn Pillow pandas numpy
```

---

## Dataset Directory Structure

All datasets must be organized as folders of class images. Each class occupies its own subdirectory under `train/` and `test/`.

### CIFAR-10 and CIFAR-100

```
cifar10/
    train/
        airplane/
        automobile/
        ...
    test/
        airplane/
        automobile/
        ...

cifar100/
    train/
        apple/
        beaver/
        ...
    test/
        apple/
        beaver/
        ...
```

### miniImageNet

Class folders are named `0` to `99` (100 classes).

```
miniImagenet/
    train/
        0/
        1/
        ...
        99/
    test/
        0/
        1/
        ...
        99/
```

### TinyImageNet (used for patch autoencoder pretraining)

Class folders are named `0` to `199` (200 classes).

```
tinyImagenet/
    train/
        0/
        1/
        ...
        199/
    test/
        0/
        1/
        ...
        199/
```

### 5-Datasets Benchmark (MNIST + notMNIST + SVHN + Fashion-MNIST + CIFAR-10)

All five datasets are merged into a single directory. Since notMNIST shares digit class names with MNIST, rename notMNIST class folders as `n0`, `n1`, ..., `n9`. Fashion-MNIST classes use their full names. The merged structure is:

```
five_data/
    train/
        0/          ← MNIST digits
        1/
        ...
        9/
        n0/         ← notMNIST letters (renamed)
        n1/
        ...
        n9/
        A/          ← notMNIST (letters A–J)
        B/
        ...
        J/
        Tshirt_top/ ← Fashion-MNIST
        Trouser/
        Pullover/
        Dress/
        Coat/
        Sandal/
        Shirt/
        Sneaker/
        Bag/
        Ankle_boot/
        airplane/   ← CIFAR-10
        automobile/
        bird/
        cat/
        deer/
        dog/
        frog/
        horse/
        ship/
        truck/
    test/
        (same structure as train/)
```

---

## Experiments

The `--experiment` flag selects the Stage-2 continual benchmark. The IDs below match `args.py`:

| ID | Benchmark | Tasks | Classes/Task |
|----|-----------|-------|-------------|
| 1  | CIFAR-10 | 5 | 2 |
| 2  | CIFAR-100 | 20 | 5 |
| 3  | miniImageNet (20-split) | 20 | 5 |
| 4  | miniImageNet (10-split) | 10 | 10 |
| 5  | 5-Datasets (MNIST + notMNIST + SVHN + Fashion-MNIST + CIFAR-10) | 5 | 10 |

> **Note:** TinyImageNet is used exclusively as the external pretraining dataset for CNN-PatchAE and ViT-PatchAE (Stage 1). It is loaded automatically when `--learn_side` is specified and does not require setting `--experiment`. The `--experiment` flag only applies to Stage 2 continual training and evaluation.

---

## Training and Evaluation

Create a checkpoints directory before running:

```bash
mkdir checkpoints
```

The entry point is `main.py`. A single run performs Stage 1 (`--learn_side`), Stage 2 (`--learn_global`), and/or evaluation (`--eval`) depending on the flags provided.

### Stage 1 — Patch Autoencoder Pretraining

#### CNN-PatchAE

```bash
python main.py \
  --model cnn \
  --learn_side \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_side 20 \
  --lr_side 1e-4 \
  --feat_map 64 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### ViT-PatchAE

```bash
python main.py \
  --model vit \
  --learn_side \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_side 20 \
  --lr_side 1e-4 \
  --vit_patch 4 \
  --vit_dim 128 \
  --vit_heads 4 \
  --n_blocks 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

Patch autoencoder checkpoints are saved as `cnn_p16_side{i}.pth` or `vit_p16_side{i}.pth` in the checkpoints directory. Pretraining runs once on TinyImageNet and the same checkpoints are reused for all downstream experiments. When running the multi-run experiments below, point every run at these shared checkpoints with `--side_checkpoints` (see the *Multi-Run Experiments* section).

---

### Stage 2 — Continual Global Model Training

#### CNN-ImageAE on CIFAR-10 (Experiment 1)

```bash
python main.py \
  --model cnn \
  --experiment 1 \
  --learn_global \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 30 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 10 \
  --proto_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### ViT-ImageAE on CIFAR-10 (Experiment 1)

```bash
python main.py \
  --model vit \
  --experiment 1 \
  --learn_global \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 30 \
  --lr_glob 1e-4 \
  --vit_patch 4 \
  --vit_dim 128 \
  --vit_heads 4 \
  --n_blocks 3 \
  --ewc_lambda 10 \
  --proto_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.5 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN-ImageAE on CIFAR-100 (Experiment 2)

```bash
python main.py \
  --model cnn \
  --experiment 2 \
  --learn_global \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 30 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 10 \
  --proto_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN-ImageAE on miniImageNet (Experiment 3)

```bash
python main.py \
  --model cnn \
  --experiment 3 \
  --learn_global \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 100 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 10 \
  --proto_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

---

### Evaluation — Linear Classifier (CIL and TIL)

#### CNN — CIL evaluation

```bash
python main.py \
  --model cnn \
  --experiment 1 \
  --eval \
  --cil \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN — TIL evaluation

```bash
python main.py \
  --model cnn \
  --experiment 1 \
  --eval \
  --til \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### ViT — CIL evaluation

```bash
python main.py \
  --model vit \
  --experiment 1 \
  --eval \
  --cil \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --vit_patch 4 \
  --vit_dim 128 \
  --vit_heads 4 \
  --n_blocks 3 \
  --lr_decay_factor 0.5 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### Forgetting evaluation (per-task accuracy after each task)

Add `--eval_forget` to any CIL evaluation command:

```bash
python main.py \
  --model cnn \
  --experiment 1 \
  --eval \
  --cil \
  --eval_forget \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

---

## Multi-Run Experiments (seeds and class orders)

To measure robustness, `run_experiments.py` wraps `main.py` and repeats a full Stage-2 + evaluation run multiple times, writing one JSON result file per run plus a summary (mean ± std) into `--results_dir`. It accepts every `main.py` argument, plus a `--run_mode` selector:

| `--run_mode` | Behaviour |
|---|---|
| `single` | One run (equivalent to calling `main.py` directly). |
| `seed_fixed_order` | Repeats the run with different random seeds at a fixed class order (15 runs for CIFAR-10/CIFAR-100, 5 for miniImageNet/5-Datasets). |
| `class_order` | 5 runs, each with a different random class-to-task assignment at a fixed seed. Available for experiments 1, 2, 3, and 5. |

The random seed controls Stage-2 weight initialization, batch ordering, and GFC/EWC sampling. **Stage-1 patch pretraining is never reseeded**, so run Stage 1 once and share those checkpoints across all runs with `--side_checkpoints`.

#### Seed runs — CIFAR-10, CNN (15 runs)

```bash
python run_experiments.py \
  --model cnn \
  --experiment 1 \
  --run_mode seed_fixed_order \
  --learn_global --eval --cil \
  --n_parts 16 --img_size 64 --batch 128 \
  --epochs_glob 30 --lr_glob 1e-4 \
  --epochs_lin 150 --lr_lin 1e-4 \
  --feat_map 64 --ewc_lambda 10 \
  --proto_alpha 0.5 --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints \
  --side_checkpoints ./checkpoints \
  --results_dir ./results
```

#### Seed runs — CIFAR-10, ViT (15 runs)

```bash
python run_experiments.py \
  --model vit \
  --experiment 1 \
  --run_mode seed_fixed_order \
  --learn_global --eval --cil \
  --n_parts 16 --img_size 64 --batch 128 \
  --epochs_glob 30 --lr_glob 1e-4 --lr_decay_factor 0.5 \
  --epochs_lin 150 --lr_lin 1e-4 \
  --vit_patch 4 --vit_dim 128 --vit_heads 4 --n_blocks 3 \
  --ewc_lambda 10 --proto_alpha 0.5 --n_repeat_layer_adapter 3 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints \
  --side_checkpoints ./checkpoints \
  --results_dir ./results
```

#### Class-order runs — CIFAR-100, CNN (5 runs)

```bash
python run_experiments.py \
  --model cnn \
  --experiment 2 \
  --run_mode class_order \
  --learn_global --eval --cil \
  --n_parts 16 --img_size 64 --batch 128 \
  --epochs_glob 30 --lr_glob 1e-4 \
  --epochs_lin 150 --lr_lin 1e-4 \
  --feat_map 64 --ewc_lambda 10 \
  --proto_alpha 0.5 --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints \
  --side_checkpoints ./checkpoints \
  --results_dir ./results
```


---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| --model | cnn | Model: cnn or vit |
| --experiment | 1 | Benchmark ID (see table above) |
| --ewc_lambda | 10 | EWC penalty weight λ |
| --proto_alpha | 0.1 | GFC / prototype drift loss weight |
| --n_parts | 16 | Number of patches (4×4 grid) |
| --img_size | 64 | Input image size |
| --epochs_side | 20 | Epochs for patch autoencoder pretraining |
| --epochs_glob | 1 | Epochs per task for global model |
| --epochs_lin | 1 | Epochs for linear classifier |
| --n_repeat_layer_adapter | 3 | Adapter layers per depth |
| --feat_map | 64 | [CNN] Base feature width F |
| --vit_dim | 128 | [ViT] Token embedding dimension |
| --vit_patch | 4 | [ViT] Sub-token size in pixels |
| --n_blocks | 3 | [ViT] Number of Transformer encoder blocks |
| --cil | off | Evaluate under CIL protocol |
| --til | off | Evaluate under TIL protocol |
| --eval_forget | off | Enable per-task forgetting measurement |
| --seed | None | Random seed for Stage-2 (init, batch order, GFC/EWC sampling) |
| --run_mode | single | `run_experiments.py` mode: single / seed_fixed_order / class_order |
| --fast_cnn | off | Faster CNN training path (same architecture and results) |
| --results_dir | ./results | Directory for per-run JSON results and summary statistics |
| --checkpoints | ./checkpoints | Directory for Stage-2 checkpoints |
| --side_checkpoints | =checkpoints | Directory of shared Stage-1 patch checkpoints |
| --data_root | ./ | Root directory for datasets |

> **Note:** `--epochs_glob` and `--epochs_lin` default to `1` in `args.py` for quick smoke tests. Set them explicitly (e.g. `--epochs_glob 30 --epochs_lin 150`) for full training, as in the commands above.

---

## Checkpoint Files

After training, the following files are saved in `--checkpoints`:

| File | Description |
|---|---|
| cnn_p16_side{i}.pth | CNN-PatchAE branch for patch position i |
| vit_p16_side{i}.pth | ViT-PatchAE branch for patch position i |
| GlobalDecoder_exp{e}_p16_task{t}.pth | CNN-ImageAE after task t |
| GlobalViT_exp{e}_p16_task{t}.pth | ViT-ImageAE after task t |
| GFC_exp{e}_p16_task{t}_d{d}.pth | GFC centroid at depth d after task t |
| cilclassifier_cnn_exp{e}_p16_task{t}.pth | CNN CIL linear classifier after task t |
| tilclassifier_cnn_exp{e}_p16_task{t}.pth | CNN TIL linear classifier after task t |
| cilclassifier_vit_exp{e}_p16_task{t}.pth | ViT CIL linear classifier after task t |
| tilclassifier_vit_exp{e}_p16_task{t}.pth | ViT TIL linear classifier after task t |

Per-run results from `run_experiments.py` are written to `--results_dir` as JSON files (one per seed or class order) together with a summary reporting mean ± standard deviation across runs.

**./results** folder contains json experimental results across seeds and class orders.
