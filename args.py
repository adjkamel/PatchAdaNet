

import argparse
import json
import sys
from pathlib import Path

#----------------------- Build and return the argparse parser with all CLI options.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CNN / ViT continual learning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file. Keys match argument names (without --)."
    )

    p.add_argument(
        "--model", type=str, choices=["cnn", "vit"], default="cnn",
        help="Which backbone to use: 'cnn' (PatchNet) or 'vit' (SideViT)."
    )

    p.add_argument(
        "--experiment", type=int, default=1,
        help=(
            "Which continual-learning benchmark to run: "
            "1=CIFAR-10, 2=CIFAR-100, 3=miniImageNet(20-split), "
            "4=miniImageNet(10-split), 5=5Datasets"
        )
    )

    p.add_argument("--prototype", action="store_true", default=False, help=" in case no prototype.")

    p.add_argument("--sub_perc", type=float, default=0.25, help="Percentage of the dataset to use.")

    p.add_argument("--supervised", action="store_true", default=False, help="Use supervised learning.")

    p.add_argument("--n_parts",    type=int,   default=16,     help="Number of patches (perfect square, e.g. 16 → 4×4 grid).")
    p.add_argument("--img_size",   type=int,   default=64,     help="All images resized to this size (px).")
    p.add_argument("--batch",      type=int,   default=128,    help="Batch size.")
    p.add_argument("--epochs_side",type=int,   default=20,     help="Epochs for Stage-1 (side-net / SideViT) training.")
    p.add_argument("--epochs_glob",type=int,   default=1,     help="Epochs for Stage-2 (GlobalDecoder / GlobalViTAE) per task.")
    p.add_argument("--epochs_lin", type=int,   default=1,    help="Epochs for linear-probe evaluation.")
    p.add_argument("--lr_side",    type=float, default=1e-4,   help="Learning rate for Stage-1.")
    p.add_argument("--lr_glob",    type=float, default=1e-4,   help="Learning rate for Stage-2.")
    p.add_argument("--lr_lin",     type=float, default=1e-4,   help="Learning rate for linear probe.")
    p.add_argument("--lr_decay_factor",     type=float, default=0.8,   help="Learning rate decay factor for ORPlateau.")
    p.add_argument("--ewc_lambda", type=float, default=5000, help="EWC penalty weight.")
    p.add_argument("--proto_alpha", type=float, default=0.1, help="prototype weight.")
    p.add_argument("--n_repeat_layer_adapter", type=int, default=3, help="number of layers in the adapter blocks of the ViT or CNN  model.")
    p.add_argument("--n_shot", type=int, default=None, help="Number of shots for few-shot learning.")
    p.add_argument("--eval_forget", action="store_true", default=False, help=" per task previous task eval , for forgetiing")
    p.add_argument("--train_sidenet_global", action="store_true", default=False, help="Train side network together with global model (end-to-end).")

    p.add_argument("--cont_train_cil", action="store_true", default=False,
                   help="[ADD] Resume CIL linear-classifier evaluation from --cont_epoch task.")
    p.add_argument("--cont_train_til", action="store_true", default=False,
                   help="[ADD] Resume TIL linear-classifier evaluation from --cont_epoch task.")
    p.add_argument("--cont_epoch", type=int, default=1,
                   help="[ADD] Task id to resume from when --cont_train_cil or --cont_train_til is set.")

    p.add_argument("--feat_map",   type=int,   default=64,
                   help="[CNN] Feature-map width for PatchNet / GlobalDecoder.")

    p.add_argument("--vit_patch",  type=int,   default=4,
                   help="[ViT] Token size (pixels) inside each side patch.")
    p.add_argument("--vit_dim",    type=int,   default=128,
                   help="[ViT] Token embedding dimension.")
    p.add_argument("--vit_heads",  type=int,   default=4,
                   help="[ViT] Number of attention heads.")
    p.add_argument("--n_blocks",   type=int,   default=3,
                   help="[ViT] Transformer blocks in SideViT (= depth levels in GlobalViTAE).")

    p.add_argument("--learn_side",  action="store_true", default=False,
                   help="Run Stage-1 (train side-nets / SideViTs).")
    p.add_argument("--learn_global",action="store_true", default=False,
                   help="Run Stage-2 (train GlobalDecoder / GlobalViTAE continually).")
    p.add_argument("--eval",        action="store_true", default=True,
                   help="Run evaluation after training.")
    p.add_argument("--cil",         action="store_true", default=False,
                   help="Evaluate in Class-Incremental Learning (CIL) mode.")
    p.add_argument("--til",         action="store_true", default=False,
                   help="Evaluate in Task-Incremental Learning (TIL) mode.")

    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for Stage-2 training (weight init, batch ordering, "
                        "GFC/EWC sampling). Stage-1 patch pretraining is never reseeded. "
                        "If None, no seed is applied.")
    p.add_argument("--fast_cnn", action="store_true", default=False,
                   help="Use cnn_traintest_fast instead of cnn_traintest for CNN training. "
                        "The fast version pre-caches classifier features, reduces EWC Fisher batches "
                        "from 50 to 5, and reuses the frozen conv cache across tasks. "
                        "Model architecture, parameters, and training strategy are unchanged.")
    p.add_argument("--run_mode", type=str,
                   choices=["single", "seed_fixed_order", "class_order"],
                   default="single",
                   help="single: one run | seed_fixed_order: N runs with different seeds "
                        "(15 for CIFAR-10/100, 5 for miniImageNet/5-Datasets) | "
                        "class_order: 5 runs with shuffled class-to-task assignment.")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Directory for per-run JSON result files and summary statistics.")

    p.add_argument("--checkpoints", type=str, default="./checkpoints",
                   help="Directory for Stage-2 checkpoints (global model, classifiers, GFC).")
    p.add_argument("--side_checkpoints", type=str, default=None,
                   help="Directory containing shared Stage-1 patch autoencoder checkpoints "
                        "(CNN-PatchAE / ViT-PatchAE). If not set, defaults to --checkpoints. "
                        "Set this when running multi-seed or class-order experiments so all "
                        "runs share the same pretrained patch encoders.")
    p.add_argument("--data_root",   type=str, default="./",
                   help="Root directory for datasets.")

    return p

#----------------------- Load a JSON config file and return it as a plain dict.
def load_config_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

#----------------------- Parse arguments with priority CLI > JSON config > argparse defaults.
def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()

    pre, _ = parser.parse_known_args(argv)

    if pre.config is not None:
        cfg_dict = load_config_file(pre.config)

        parser.set_defaults(**cfg_dict)

    args = parser.parse_args(argv)

    if not args.learn_side and not args.learn_global and not args.eval:
        args.eval = True

    if args.side_checkpoints is None:
        args.side_checkpoints = args.checkpoints

    return args

if __name__ == "__main__":
    a = parse_args()
    print("Parsed arguments:")
    for k, v in sorted(vars(a).items()):
        print(f"  {k:20s} = {v}")
