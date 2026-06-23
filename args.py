import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    # Argument parser for PatchAdaNet CNN/ViT continual learning pipeline
    p = argparse.ArgumentParser(
        description="PatchAdaNet: CNN / ViT continual learning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON config file.")

    p.add_argument("--model", type=str, choices=["cnn", "vit"], default="cnn",
                   help="Backbone: cnn (CNN-PatchAE/CNN-ImageAE) or vit (ViT-PatchAE/ViT-ImageAE).")

    p.add_argument("--experiment", type=int, default=1,
                   help="Benchmark: 1=5Splits-CIFAR-10, 2=20SplitsCIFAR-100,  3=20Split-miniImageNet, 4=10Split-miniImageNet, 5= 5Datasets, ")

    p.add_argument("--gfc", action="store_true", default=False,
                   help="Enable Global Feature Centroid (GFC) regularization.")

    p.add_argument("--sub_perc", type=float, default=0.25,
                   help="Fraction of dataset to use.")

    p.add_argument("--n_parts",    type=int,   default=16,
                   help="Number of patches (perfect square, e.g. 16 for 4x4 grid).")
    p.add_argument("--img_size",   type=int,   default=64,
                   help="Input image size (pixels).")
    p.add_argument("--batch",      type=int,   default=128,
                   help="Batch size.")
    p.add_argument("--epochs_side",type=int,   default=20,
                   help="Epochs for CNN-PatchAE or ViT-PatchAE pretraining.")
    p.add_argument("--epochs_glob",type=int,   default=30,
                   help="Epochs for CNN-ImageAE or ViT-ImageAE per task.")
    p.add_argument("--epochs_lin", type=int,   default=150,
                   help="Epochs for linear classifier training.")
    p.add_argument("--lr_side",    type=float, default=1e-4,
                   help="Learning rate for patch autoencoder pretraining.")
    p.add_argument("--lr_glob",    type=float, default=1e-4,
                   help="Learning rate for continual global model training.")
    p.add_argument("--lr_lin",     type=float, default=1e-4,
                   help="Learning rate for linear classifier.")
    p.add_argument("--lr_decay_factor", type=float, default=0.8,
                   help="ReduceLROnPlateau decay factor.")
    p.add_argument("--ewc_lambda", type=float, default=5000,
                   help="EWC penalty weight (lambda).")
    p.add_argument("--gfc_alpha",  type=float, default=0.1,
                   help="GFC drift loss weight (alpha).")
    p.add_argument("--n_repeat_layer_adapter", type=int, default=3,
                   help="Number of adapter layers per depth in CNN-ImageAE or ViT-ImageAE.")
    p.add_argument("--n_shot", type=int, default=None,
                   help="Shots per class for few-shot evaluation.")
    p.add_argument("--eval_forget", action="store_true", default=False,
                   help="Evaluate per-task forgetting after each continual task.")
    p.add_argument("--train_sidenet_global", action="store_true", default=False,
                   help="Fine-tune patch encoders jointly with the global model.")

    p.add_argument("--cont_train_cil", action="store_true", default=False,
                   help="Resume CIL classifier evaluation from --cont_epoch.")
    p.add_argument("--cont_train_til", action="store_true", default=False,
                   help="Resume TIL classifier evaluation from --cont_epoch.")
    p.add_argument("--cont_epoch", type=int, default=1,
                   help="Task index to resume from when using --cont_train_cil/til.")

    p.add_argument("--feat_map",   type=int,   default=64,
                   help="[CNN] Base feature width F for CNN-PatchAE and CNN-ImageAE.")

    p.add_argument("--vit_patch",  type=int,   default=4,
                   help="[ViT] Sub-token size (pixels) within each patch.")
    p.add_argument("--vit_dim",    type=int,   default=128,
                   help="[ViT] Token embedding dimension.")
    p.add_argument("--vit_heads",  type=int,   default=4,
                   help="[ViT] Number of attention heads.")
    p.add_argument("--n_blocks",   type=int,   default=3,
                   help="[ViT] Number of Transformer encoder blocks (encoding depth D).")

    p.add_argument("--learn_side",   action="store_true", default=False,
                   help="Run Stage 1: pretrain CNN-PatchAE or ViT-PatchAE.")
    p.add_argument("--learn_global", action="store_true", default=False,
                   help="Run Stage 2: train CNN-ImageAE or ViT-ImageAE continually.")
    p.add_argument("--eval",         action="store_true", default=True,
                   help="Run linear classifier evaluation.")
    p.add_argument("--cil",          action="store_true", default=False,
                   help="Evaluate under Class-Incremental Learning (CIL) protocol.")
    p.add_argument("--til",          action="store_true", default=False,
                   help="Evaluate under Task-Incremental Learning (TIL) protocol.")

    p.add_argument("--checkpoints", type=str, default="./checkpoints",
                   help="Directory for saving model checkpoints.")
    p.add_argument("--data_root",   type=str, default="./",
                   help="Root directory containing datasets.")

    return p


def load_config_file(path: str) -> dict:
    # Load JSON config and return as dict
    with open(path) as f:
        return json.load(f)


def parse_args(argv=None) -> argparse.Namespace:
    # Parse arguments with priority: CLI > JSON config > defaults
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        cfg_dict = load_config_file(pre.config)
        parser.set_defaults(**cfg_dict)
    args = parser.parse_args(argv)
    if not args.learn_side and not args.learn_global and not args.eval:
        args.eval = True
    return args


if __name__ == "__main__":
    a = parse_args()
    print("PatchAdaNet arguments:")
    for k, v in sorted(vars(a).items()):
        print(f"  {k:30s} = {v}")