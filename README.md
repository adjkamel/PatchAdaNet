# PatchAdaNet

**PatchAdaNet: Unsupervised Continual Learning via Independent Patch-Level Representations**

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

| ID | Benchmark | Tasks | Classes/Task |
|----|-----------|-------|-------------|
| 1  | 5-Split CIFAR-10 | 5 | 2 |
| 2  | 20-Split CIFAR-100 | 20 | 5 |
| 3  | 20-Split miniImageNet | 20 | 5 |
| 4  | 10-Split miniImageNet | 10 | 10 |
| 5  | 5-Datasets (MNIST + notMNIST + SVHN + Fashion-MNIST + CIFAR-10) | 5 | 10 |

> **Note:** TinyImageNet is used exclusively as the external pretraining dataset for CNN-PatchAE and ViT-PatchAE (Stage 1). It is loaded automatically when `--learn_side` is specified and does not require setting `--experiment`. The `--experiment` flag only applies to Stage 2 continual training and evaluation.

---

## Training and Evaluation

Create a checkpoints directory before running:

```bash
mkdir checkpoints
```

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

Patch autoencoder checkpoints are saved as `cnn_p16_side{i}.pth` or `vit_p16_side{i}.pth` in the checkpoints directory. Pretraining runs once on TinyImageNet and the same checkpoints are reused for all downstream experiments.

---

### Stage 2 — Continual Global Model Training

#### CNN-ImageAE on 5-Split CIFAR-10 (Experiment 5)

```bash
python main.py \
  --model cnn \
  --experiment 5 \
  --learn_global \
  --gfc \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 30 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 5000 \
  --gfc_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### ViT-ImageAE on 5-Split CIFAR-10 (Experiment 5)

```bash
python main.py \
  --model vit \
  --experiment 5 \
  --learn_global \
  --gfc \
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
  --gfc_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.5 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN-ImageAE on 20-Split CIFAR-100 (Experiment 6)

```bash
python main.py \
  --model cnn \
  --experiment 6 \
  --learn_global \
  --gfc \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 30 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 5000 \
  --gfc_alpha 0.5 \
  --n_repeat_layer_adapter 3 \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN-ImageAE on 20-Split miniImageNet (Experiment 4)

```bash
python main.py \
  --model cnn \
  --experiment 4 \
  --learn_global \
  --gfc \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_glob 100 \
  --lr_glob 1e-4 \
  --feat_map 64 \
  --ewc_lambda 5000 \
  --gfc_alpha 0.5 \
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
  --experiment 5 \
  --eval \
  --cil \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --gfc \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### CNN — TIL evaluation

```bash
python main.py \
  --model cnn \
  --experiment 5 \
  --eval \
  --til \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --gfc \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### ViT — CIL evaluation

```bash
python main.py \
  --model vit \
  --experiment 5 \
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
  --gfc \
  --lr_decay_factor 0.5 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

#### Forgetting evaluation (per-task accuracy after each task)

Add `--eval_forget` to any CIL evaluation command:

```bash
python main.py \
  --model cnn \
  --experiment 5 \
  --eval \
  --cil \
  --eval_forget \
  --n_parts 16 \
  --img_size 64 \
  --batch 128 \
  --epochs_lin 150 \
  --lr_lin 1e-4 \
  --feat_map 64 \
  --gfc \
  --lr_decay_factor 0.8 \
  --data_root /path/to/data \
  --checkpoints ./checkpoints
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `cnn` | Backbone: `cnn` or `vit` |
| `--experiment` | `5` | Benchmark ID (see table above) |
| `--gfc` | off | Enable GFC drift regularization |
| `--ewc_lambda` | `5000` | EWC penalty weight λ |
| `--gfc_alpha` | `0.1` | GFC drift loss weight α |
| `--n_parts` | `16` | Number of patches (4×4 grid) |
| `--img_size` | `64` | Input image size |
| `--epochs_side` | `20` | Epochs for patch autoencoder pretraining |
| `--epochs_glob` | `30` | Epochs per task for global model |
| `--epochs_lin` | `150` | Epochs for linear classifier |
| `--n_repeat_layer_adapter` | `3` | Adapter layers per depth |
| `--feat_map` | `64` | [CNN] Base feature width F |
| `--vit_dim` | `128` | [ViT] Token embedding dimension |
| `--vit_patch` | `4` | [ViT] Sub-token size in pixels |
| `--n_blocks` | `3` | [ViT] Number of Transformer encoder blocks |
| `--cil` | off | Evaluate under CIL protocol |
| `--til` | off | Evaluate under TIL protocol |
| `--eval_forget` | off | Enable per-task forgetting measurement |
| `--checkpoints` | `./checkpoints` | Directory for saving checkpoints |
| `--data_root` | `./` | Root directory for datasets |

---

## Checkpoint Files

After training, the following files are saved in `--checkpoints`:

| File | Description |
|------|-------------|
| `cnn_p16_side{i}.pth` | CNN-PatchAE branch for patch position `i` |
| `vit_p16_side{i}.pth` | ViT-PatchAE branch for patch position `i` |
| `GlobalDecoder_exp{e}_p16_task{t}.pth` | CNN-ImageAE after task `t` |
| `GlobalViT_exp{e}_p16_task{t}.pth` | ViT-ImageAE after task `t` |
| `GFC_exp{e}_p16_task{t}_d{d}.pth` | GFC centroid at depth `d` after task `t` |
| `cilclassifier_cnn_exp{e}_p16_task{t}.pth` | CNN CIL linear classifier after task `t` |
| `tilclassifier_cnn_exp{e}_p16_task{t}.pth` | CNN TIL linear classifier after task `t` |
| `cilclassifier_vit_exp{e}_p16_task{t}.pth` | ViT CIL linear classifier after task `t` |
| `tilclassifier_vit_exp{e}_p16_task{t}.pth` | ViT TIL linear classifier after task `t` |


