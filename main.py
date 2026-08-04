import sys
import importlib
import torch
from pathlib import Path

from args import parse_args
from dataloader import create_continual_loaders

_use_fast_cnn = '--fast_cnn' in sys.argv
_cnn_module_name = 'cnn_traintest_fast' if _use_fast_cnn else 'cnn_traintest'
_cnn_mod = importlib.import_module(_cnn_module_name)
train_sidenets        = _cnn_mod.train_sidenets
load_all_sides        = _cnn_mod.load_all_sides
train_global_decoder  = _cnn_mod.train_global_decoder
cnn_evaluate_classifier = _cnn_mod.cnn_evaluate_classifier
build_lut             = _cnn_mod.build_lut
print(f"[CNN module] Using: {_cnn_module_name}")

from cnn_models import PatchNetMulti, GlobalDecoder

from vit_models import SideViTMulti, GlobalViTAE
from vit_traintest import (
    eval_linear_classifier,
    train_sidevits,
    load_all_sidevits,
    train_global_vit,
    vit_evaluate_classifier,
)


#----------------------- Return the task dictionary for the chosen experiment.
def get_tasks(experiment: int, img_size: int, data_root: str) -> dict:

    if experiment == 1:
        names = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
        dc  = {n: i for i, n in enumerate(names)}
        itc = {v: k for k, v in dc.items()}
        return {
            i + 1: {
                "name": "cifar10", "img_size": img_size,
                "selected_classes": {itc[j]: j for j in range(i * 2, (i + 1) * 2)},
                "permute": False, "train": True, "grayscale": False, "root": data_root,
            }
            for i in range(5)
        }

    if experiment == 2:

        names = [
            "beaver", "dolphin", "otter", "seal", "whale",
            "aquarium_fish", "flatfish", "ray", "shark", "trout",
            "orchid", "poppy", "rose", "sunflower", "tulip",
            "bottle", "bowl", "can", "cup", "plate",
            "apple", "mushroom", "orange", "pear", "sweet_pepper",
            "clock", "keyboard", "lamp", "telephone", "television",
            "bed", "chair", "couch", "table", "wardrobe",
            "bee", "beetle", "butterfly", "caterpillar", "cockroach",
            "bear", "leopard", "lion", "tiger", "wolf",
            "bridge", "castle", "house", "road", "skyscraper",
            "cloud", "forest", "mountain", "plain", "sea",
            "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
            "fox", "porcupine", "possum", "raccoon", "skunk",
            "crab", "lobster", "snail", "spider", "worm",
            "baby", "boy", "girl", "man", "woman",
            "crocodile", "dinosaur", "lizard", "snake", "turtle",
            "hamster", "mouse", "rabbit", "shrew", "squirrel",
            "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree",
            "bicycle", "bus", "motorcycle", "pickup_truck", "train",
            "lawn_mower", "rocket", "streetcar", "tank", "tractor",
        ]
        dc  = {n: i for i, n in enumerate(names)}
        itc = {v: k for k, v in dc.items()}
        return {
            i + 1: {
                "name": "cifar100", "img_size": img_size,
                "selected_classes": {itc[j]: j for j in range(i * 5, (i + 1) * 5)},
                "permute": False, "train": True, "grayscale": False, "root": data_root,
            }
            for i in range(20)
        }
    if experiment == 3:
        itc = {i: str(i) for i in range(100)}
        return {
            i + 1: {
                "name": "miniImagenet", "img_size": img_size,
                "selected_classes": {itc[j]: j for j in range(i * 5, (i + 1) * 5)},
                "permute": False, "train": True, "grayscale": False, "root": data_root,
            }
            for i in range(20)
        }

    if experiment == 4:
        itc = {i: str(i) for i in range(100)}
        return {
            i + 1: {
                "name": "miniImagenet", "img_size": img_size,
                "selected_classes": {itc[j]: j for j in range(i * 10, (i + 1) * 10)},
                "permute": False, "train": True, "grayscale": False, "root": data_root,
            }
            for i in range(10)
        }

    if experiment == 5:
        all_names = (
            [str(i) for i in range(10)] + [f"n{i}" for i in range(10)] +
            list("ABCDEFGHIJ") +
            ["Tshirt_top", "Trouser", "Pullover", "Dress", "Coat",
             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"] +
            ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]
        )
        dc  = {n: i for i, n in enumerate(all_names)}
        itc = {v: k for k, v in dc.items()}
        return {
            i + 1: {
                "name": "five_data", "img_size": img_size,
                "selected_classes": {itc[j]: j for j in range(i * 10, (i + 1) * 10)},
                "permute": False, "train": True, "grayscale": False, "root": data_root,
            }
            for i in range(5)
        }

    raise ValueError(f"Unknown experiment id: {experiment}")

#----------------------- Wrap create_continual_loaders for a task config.
def make_loader(args, cfg: dict, train: bool, batch: int,n_shot=None):
    cfg = dict(cfg)
    cfg["train"] = train
    return create_continual_loaders(cfg=cfg, sub_perc=args.sub_perc,batch_size=batch, shuffle=train, n_shot=n_shot)

#----------------------- Construct the multi-patch CNN encoder (PatchNetMulti).
def build_sidenet(args, grayscale=False):
    ch = 1 if grayscale else 3
    return PatchNetMulti(
        ch, args.feat_map, n_parts=args.n_parts, exp=args.experiment
    ).to(args.device)

#----------------------- Construct the CNN GlobalDecoder for the current experiment.
def build_global_dec(args,num_classes):
    return GlobalDecoder(
        3, args.feat_map, n_parts=args.n_parts,
        exp=args.experiment, img_size=args.img_size,n_layer_adapter=args.n_repeat_layer_adapter, use_proto=args.prototype, supervised=args.supervised,num_classes=num_classes
    ).to(args.device)

#----------------------- Return the GlobalDecoder checkpoint path for a task.
def globdec_ckpt(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_GlobalDecoder_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Return the CNN CIL classifier checkpoint path for a task.
def cilclassifier_ckptcnn(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_cilclassifier_cnn_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Return the ViT CIL classifier checkpoint path for a task.
def cilclassifier_ckptvit(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_cilclassifier_vit_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Return the CNN TIL classifier checkpoint path for a task.
def tilclassifier_ckptcnn(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_tilclassifier_cnn_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Return the ViT TIL classifier checkpoint path for a task.
def tilclassifier_ckptvit(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_tilclassifier_vit_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Return the CNN prototype checkpoint path for a task and depth.
def cnnproto_ckpt(args, p,task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_CNNproto_exp{args.experiment}_p{args.n_parts}_task{task_id}_{p}.pth"
    )

#----------------------- Return the ViT prototype checkpoint path for a task.
def vitproto_ckpt(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_VITproto_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Construct the multi-patch ViT encoder (SideViTMulti).
def build_sidevit(args, grayscale=False):
    ch = 1 if grayscale else 3
    return SideViTMulti(
        channel=ch, img_h=args.img_size, img_w=args.img_size,
        n_parts=args.n_parts, patch_size=args.vit_patch,
        dim=args.vit_dim, n_blocks=args.n_blocks,
        heads=args.vit_heads, exp=args.experiment,
    ).to(args.device)

#----------------------- Construct the ViT GlobalViTAE for the current experiment.
def build_global_vit(args,num_classes):
    return GlobalViTAE(
        channel=3, img_h=args.img_size, img_w=args.img_size,
        n_parts=args.n_parts, patch_size=args.vit_patch,
        dim=args.vit_dim, n_blocks=args.n_blocks,
        heads=args.vit_heads, exp=args.experiment,n_layer_adapter=args.n_repeat_layer_adapter,use_proto=args.prototype, supervised=args.supervised,num_classes=num_classes
    ).to(args.device)

#----------------------- Return the GlobalViTAE checkpoint path for a task.
def globvit_ckpt(args, task_id: int) -> Path:
    return (
        Path(args.checkpoints)
        / f"{args.supervised}_GlobalViT_exp{args.experiment}_p{args.n_parts}_task{task_id}.pth"
    )

#----------------------- Entry point: run Stage 1, Stage 2, and/or evaluation per the CLI flags.
def main():

    args = parse_args()

    if getattr(args, "seed", None) is not None and not args.learn_side:
        import random, numpy as np
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"  [Seed] Stage-2 seed set to {seed}")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)

    tasks = get_tasks(args.experiment, args.img_size, args.data_root)

    ''''
    log_dir = build_log_path(args, tasks[1]["name"] if tasks else "dataset").parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = build_log_path(args, tasks[1]["name"] if tasks else "dataset")

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Logging to: {log_path}")
    print("=" * 80)
    '''
    print("Arguments:")
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")
    print("=" * 80)
    print()

    num_tasks = len(tasks)

    sidenet_cfg = {
        "name":             "tinyImagenet",
        "img_size":         args.img_size,
        "selected_classes": None,
        "permute":          False,
        "train":            True,
        "grayscale":        False,
        "root":             args.data_root,
    }

    print(f"\nModel: {args.model.upper()} | Mode {args.supervised} | Experiment: {args.experiment} "
          f"| N_PARTS={args.n_parts} | IMG_SIZE={args.img_size} | Tasks={num_tasks}")

    if args.model == "cnn":

        print(f"FEAT_MAP={args.feat_map} | "
              f"BATCH={args.batch} | LR_SIDE={args.lr_side} | LR_GLOB={args.lr_glob}")

        if args.learn_side:
            print("\n" + "=" * 50)
            print("STAGE 1 – PatchNet sides (one-time)")
            print("=" * 50)

            sidenet      = build_sidenet(args, grayscale=False)
            train_loader = make_loader(args, sidenet_cfg, train=True, batch=args.batch,n_shot=None)

            train_sidenets(
                model           = sidenet,
                dataloader      = train_loader,
                epochs          = args.epochs_side,
                device          = args.device,
                learn_rate      = args.lr_side,
                checkpoints_dir = args.checkpoints,
                n_parts         = args.n_parts,
                exp             = args.experiment,
                lr_decay        = args.lr_decay_factor,
            )

        if args.learn_global:
            print("\n" + "=" * 50)
            print(f"STAGE 2 – GlobalDecoder continual (Mode {args.supervised} Exp {args.experiment})")
            print("=" * 50)

            sidenet = build_sidenet(args, grayscale=False)
            sidenet = load_all_sides(
                sidenet, args.checkpoints, args.n_parts, args.experiment, args.device
            )

            ewc = None

            for task_id in sorted(tasks.keys()):
                print(f"\n  -- Task {task_id}/{num_tasks} --")

                train_loader = make_loader(args, tasks[task_id], train=True, batch=args.batch, n_shot=args.n_shot)

                global_dec = build_global_dec(args,num_classes=len(tasks[task_id]["selected_classes"]))

                prev = globdec_ckpt(args, task_id - 1)
                if prev.exists():

                    state_dict = torch.load(prev, map_location=args.device)
                    missing_keys, unexpected_keys = global_dec.load_state_dict(state_dict, strict=False)

                    if any('prototype.maps' in key for key in state_dict.keys()):
                        global_dec.prototype.is_initialised.fill_(True)

                    global_dec.invalidate_frozen_cache()
                    print(f"  Loaded GlobalDecoder from {prev}")
                    if missing_keys:
                        print(f"    Missing keys (ignored): {missing_keys}")
                    if unexpected_keys:
                        print(f"    Unexpected keys (ignored): {unexpected_keys}")

                local_class_map = {
                            gid: lidx
                            for lidx, gid in enumerate(sorted(tasks[task_id]["selected_classes"].values()))
                        }

                if args.prototype:

                    global_dec, ewc, tensor_list_proto= train_global_decoder(
                        global_dec   = global_dec,
                        sidenet      = sidenet,
                        dataloader   = train_loader,
                        epochs       = args.epochs_glob,
                        device       = args.device,
                        learn_rate   = args.lr_glob,
                        ewc          = ewc,
                        ewc_lambda   = args.ewc_lambda,
                        lr_decay     = args.lr_decay_factor,
                        lambda_proto = args.proto_alpha,
                        task_id      = task_id,
                        use_proto    = args.prototype,
                        class_map    = local_class_map,
                        supervised=args.supervised,train_sidenet=args.train_sidenet_global)

                    torch.save(tensor_list_proto[0], cnnproto_ckpt(args, "p1",task_id) )
                    torch.save(tensor_list_proto[1], cnnproto_ckpt(args, "p2",task_id) )
                    torch.save(tensor_list_proto[2], cnnproto_ckpt(args, "p3",task_id) )

                else:
                    global_dec, ewc= train_global_decoder(
                        global_dec   = global_dec,
                        sidenet      = sidenet,
                        dataloader   = train_loader,
                        epochs       = args.epochs_glob,
                        device       = args.device,
                        learn_rate   = args.lr_glob,
                        ewc          = ewc,
                        ewc_lambda   = args.ewc_lambda,
                        lr_decay     = args.lr_decay_factor,
                        lambda_proto = args.proto_alpha,
                        task_id      = task_id,
                        use_proto    = args.prototype,
                        class_map    = local_class_map,
                        supervised=args.supervised,train_sidenet=args.train_sidenet_global)

                ckpt_path = globdec_ckpt(args, task_id)
                torch.save(global_dec.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")

        if args.eval:
            print("\n" + "=" * 50)
            print("EVALUATION (CNN)")
            print("=" * 50)

            sidenet = build_sidenet(args, grayscale=False)
            sidenet = load_all_sides(
                sidenet, args.checkpoints, args.n_parts, args.experiment, args.device
            )

            clf_mode = "linear_global"

            if args.til:
                til_results = {}
                for task_id, cfg in tasks.items():
                    gckpt = globdec_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    if args.cont_train_til and task_id < args.cont_epoch:
                        print(f"    [cont_train_til] Skipping TIL task {task_id} (< cont_epoch {args.cont_epoch})")
                        continue

                    global_dec = build_global_dec(args,num_classes=None)
                    global_dec.load_state_dict(torch.load(gckpt, map_location=args.device),strict=False)

                    num_classes     = len(cfg["selected_classes"])
                    global_to_local = {
                        gid: lidx
                        for lidx, gid in enumerate(sorted(cfg["selected_classes"].values()))
                    }

                    train_loader = make_loader(args, cfg, train=True,  batch=args.batch, n_shot=None)
                    test_loader  = make_loader(args, cfg, train=False, batch=args.batch, n_shot=None)

                    ckpt_tilpath_cls = tilclassifier_ckptcnn(args, task_id)

                    if ckpt_tilpath_cls.exists():
                        print(f"    Found existing TIL classifier checkpoint for task {task_id} at {ckpt_tilpath_cls}, loading...")

                        clsfr = torch.load(ckpt_tilpath_cls, map_location=args.device)
                    else:
                        clsfr = None

                    (acc, _), clsfr = cnn_evaluate_classifier(
                        mode         = clf_mode,
                        sidenet      = sidenet,
                        global_dec   = global_dec,
                        train_loader = train_loader,
                        test_loader  = test_loader,
                        num_classes  = num_classes,
                        device       = args.device,
                        epochs_lin   = args.epochs_lin,
                        lr_lin       = args.lr_lin,
                        class_map    = global_to_local,
                        lr_decay     = args.lr_decay_factor,
                        saved=clsfr
                    )
                    til_results[task_id] = acc
                    print(f"    TIL task {task_id}: {acc:.2f}%")

                    torch.save(clsfr.state_dict(), ckpt_tilpath_cls)
                    print(f"  TIL Classfier Saved: {ckpt_tilpath_cls}")

                if til_results:
                    print(f"  Avg TIL: {sum(til_results.values()) / len(til_results):.2f}%")

            if args.cil:
                cil_results = {}
                for task_id in sorted(tasks.keys()):
                    gckpt = globdec_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    if args.cont_train_cil and task_id < args.cont_epoch:
                        print(f"    [cont_train_cil] Skipping CIL task {task_id} (< cont_epoch {args.cont_epoch})")
                        continue

                    global_dec = build_global_dec(args, num_classes=None)
                    global_dec.load_state_dict(torch.load(gckpt, map_location=args.device),strict=False)

                    seen_classes = {}

                    for t in range(1, task_id + 1):
                        seen_classes.update(tasks[t]["selected_classes"])

                    global_to_local = {
                        gid: lidx
                        for lidx, gid in enumerate(sorted(seen_classes.values()))
                    }

                    base_cfg = dict(tasks[1])
                    base_cfg["selected_classes"] = seen_classes

                    train_loader = make_loader(args, base_cfg, train=True,  batch=args.batch, n_shot=None)

                    test_loader = None if args.eval_forget else make_loader(args, base_cfg, train=False, batch=args.batch, n_shot=args.n_shot)

                    list_test_loaders = []
                    if args.eval_forget and task_id > 1:
                        for t in range(1, task_id):
                            prev_cfg = dict(tasks[t])
                            prev_cfg["selected_classes"] = dict(tasks[t]["selected_classes"])
                            prev_loader = make_loader(args, prev_cfg, train=False, batch=args.batch, n_shot=None)

                            list_test_loaders.append((prev_loader, global_to_local))

                    ckpt_cilpath_cls = cilclassifier_ckptcnn(args, task_id)

                    if ckpt_cilpath_cls.exists():
                        print(f"    Found existing CIL classifier checkpoint for task {task_id} at {ckpt_cilpath_cls}, loading...")

                        clsfr = torch.load(ckpt_cilpath_cls, map_location=args.device)
                    else:

                        clsfr=None

                    (acc, acc_forg), clsfr = cnn_evaluate_classifier(
                        mode              = clf_mode,
                        sidenet           = sidenet,
                        global_dec        = global_dec,
                        train_loader      = train_loader,
                        test_loader       = test_loader,
                        num_classes       = len(seen_classes),
                        device            = args.device,
                        epochs_lin        = args.epochs_lin,
                        lr_lin            = args.lr_lin,
                        class_map         = global_to_local,
                        lr_decay          = args.lr_decay_factor,
                        list_test_loaders = list_test_loaders,
                        saved=clsfr
                    )
                    if acc is not None:
                        cil_results[task_id] = acc
                        print(f"    CIL @ task {task_id}: {acc:.2f}%")
                    if acc_forg:
                        for t_idx, fa in enumerate(acc_forg, start=1):
                            print(f"      Forgetting task {t_idx} acc (after task {task_id}): {fa:.2f}%")

                    torch.save(clsfr, ckpt_cilpath_cls)

                if cil_results:
                    print(f"  Avg CIL: {sum(cil_results.values()) / len(cil_results):.2f}%")

    elif args.model == "vit":

        print(f"VIT_PATCH={args.vit_patch} | VIT_DIM={args.vit_dim} | "
              f"VIT_HEADS={args.vit_heads} | N_BLOCKS={args.n_blocks} | "
              f"BATCH={args.batch} | LR_SIDE={args.lr_side} | LR_GLOB={args.lr_glob}")

        if args.learn_side:
            print("\n" + "=" * 50)
            print("STAGE 1 – SideViT training ")
            print("=" * 50)

            sidevit      = build_sidevit(args, grayscale=False)
            train_loader = make_loader(args, sidenet_cfg, train=True, batch=args.batch, n_shot=None)

            train_sidevits(
                model           = sidevit,
                dataloader      = train_loader,
                epochs          = args.epochs_side,
                device          = args.device,
                learn_rate      = args.lr_side,
                checkpoints_dir = args.checkpoints,
                n_parts         = args.n_parts,
                exp             = args.experiment,
                lr_decay        = args.lr_decay_factor,
            )

        if args.learn_global:
            print("\n" + "=" * 50)
            print(f"STAGE 2 – GlobalViTAE continual (Mode {args.supervised} Exp: {args.experiment})")
            print("=" * 50)

            sidevit = build_sidevit(args, grayscale=False)
            sidevit = load_all_sidevits(
                sidevit, args.checkpoints, args.n_parts, args.experiment, args.device
            )

            ewc = None

            for task_id in sorted(tasks.keys()):
                print(f"\n  -- Task {task_id}/{num_tasks} --")
                train_loader = make_loader(args, tasks[task_id], train=True, batch=args.batch,n_shot=args.n_shot)

                global_vit = build_global_vit(args, num_classes=len(tasks[task_id]["selected_classes"]))

                prev = globvit_ckpt(args, task_id - 1)
                if prev.exists():
                    global_vit.load_state_dict(torch.load(prev, map_location=args.device))
                    global_vit.invalidate_frozen_cache()
                    print(f"  Loaded GlobalViT from {prev}")
                local_class_map = {
                        gid: lidx
                        for lidx, gid in enumerate(sorted(tasks[task_id]["selected_classes"].values()))
                    }
                if args.prototype:

                    global_vit, ewc, tensor_list_proto = train_global_vit(
                        global_vit   = global_vit,
                        sidenet      = sidevit,
                        dataloader   = train_loader,
                        epochs       = args.epochs_glob,
                        device       = args.device,
                        learn_rate   = args.lr_glob,
                        ewc          = ewc,
                        ewc_lambda   = args.ewc_lambda,
                        lr_decay     = args.lr_decay_factor,
                        lambda_proto = args.proto_alpha,
                        task_id      = task_id,
                        use_proto    = args.prototype,
                        class_map    = local_class_map,
                        supervised   = args.supervised,train_sidenet=args.train_sidenet_global)

                    torch.save(tensor_list_proto[0], cnnproto_ckpt(args, "p1",task_id) )
                    torch.save(tensor_list_proto[1], cnnproto_ckpt(args, "p2",task_id) )
                    torch.save(tensor_list_proto[2], cnnproto_ckpt(args, "p3",task_id) )

                else:
                    global_vit, ewc = train_global_vit(
                        global_vit   = global_vit,
                        sidenet      = sidevit,
                        dataloader   = train_loader,
                        epochs       = args.epochs_glob,
                        device       = args.device,
                        learn_rate   = args.lr_glob,
                        ewc          = ewc,
                        ewc_lambda   = args.ewc_lambda,
                        lr_decay     = args.lr_decay_factor,
                        lambda_proto = args.proto_alpha,
                        task_id      = task_id,
                        use_proto    = args.prototype,
                        class_map    = local_class_map,
                        supervised   = args.supervised,train_sidenet=args.train_sidenet_global)

                ckpt_path = globvit_ckpt(args, task_id)
                torch.save(global_vit.state_dict(), ckpt_path)

                print(f"  Saved: {ckpt_path}")

        if args.eval:
            print("\n" + "=" * 50)
            print("EVALUATION (ViT)")
            print("=" * 50)

            sidevit = build_sidevit(args, grayscale=False)
            sidevit = load_all_sidevits(
                sidevit, args.checkpoints, args.n_parts, args.experiment, args.device
            )

            clf_mode = "linear_global"

            if args.til:
                til_results = {}
                for task_id, cfg in tasks.items():
                    gckpt = globvit_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    if args.cont_train_til and task_id < args.cont_epoch:
                        print(f"    [cont_train_til] Skipping TIL task {task_id} (< cont_epoch {args.cont_epoch})")
                        continue

                    global_vit = build_global_vit(args, num_classes=None)
                    global_vit.load_state_dict(torch.load(gckpt, map_location=args.device),strict=False)

                    num_classes     = len(cfg["selected_classes"])
                    global_to_local = {
                        gid: lidx
                        for lidx, gid in enumerate(sorted(cfg["selected_classes"].values()))
                    }

                    train_loader = make_loader(args, cfg, train=True,  batch=args.batch,n_shot=None)
                    test_loader  = make_loader(args, cfg, train=False, batch=args.batch,n_shot=None)
                    ckpt_tilpath_cls = tilclassifier_ckptvit(args, task_id)
                    if ckpt_tilpath_cls.exists():
                        print(f"    Found existing TIL classifier checkpoint for task {task_id} at {ckpt_tilpath_cls}, loading...")
                        clsfr = torch.load(ckpt_tilpath_cls, map_location=args.device)
                    else:
                        clsfr = None

                    (acc, _), clsfr = vit_evaluate_classifier(
                        mode         = clf_mode,
                        sidenet      = sidevit,
                        global_vit   = global_vit,
                        train_loader = train_loader,
                        test_loader  = test_loader,
                        num_classes  = num_classes,
                        device       = args.device,
                        epochs_lin   = args.epochs_lin,
                        lr_lin       = args.lr_lin,
                        class_map    = global_to_local,
                        lr_decay     = args.lr_decay_factor,
                        saved=clsfr
                    )
                    til_results[task_id] = acc
                    print(f"    TIL task {task_id}: {acc:.2f}%")
                    torch.save(clsfr, ckpt_tilpath_cls)

                if til_results:
                    print(f"  Avg TIL: {sum(til_results.values()) / len(til_results):.2f}%")

            if args.cil:
                cil_results = {}
                for task_id in sorted(tasks.keys()):
                    gckpt = globvit_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    if args.cont_train_cil and task_id < args.cont_epoch:
                        print(f"    [cont_train_cil] Skipping CIL task {task_id} (< cont_epoch {args.cont_epoch})")
                        continue

                    global_vit = build_global_vit(args, num_classes=None)
                    global_vit.load_state_dict(torch.load(gckpt, map_location=args.device),strict=False)

                    seen_classes = {}
                    for t in range(1, task_id + 1):
                        seen_classes.update(tasks[t]["selected_classes"])

                    global_to_local = {
                        gid: lidx
                        for lidx, gid in enumerate(sorted(seen_classes.values()))
                    }

                    base_cfg = dict(tasks[1])
                    base_cfg["selected_classes"] = seen_classes

                    train_loader = make_loader(args, base_cfg, train=True,  batch=args.batch, n_shot=None)

                    test_loader = None if args.eval_forget else make_loader(args, base_cfg, train=False, batch=args.batch, n_shot=None)

                    list_test_loaders = []
                    if args.eval_forget and task_id > 1:
                        for t in range(1, task_id):
                            prev_cfg = dict(tasks[t])
                            prev_cfg["selected_classes"] = dict(tasks[t]["selected_classes"])
                            prev_loader = make_loader(args, prev_cfg, train=False, batch=args.batch, n_shot=None)

                            list_test_loaders.append((prev_loader, global_to_local))
                    ckpt_cilpath_cls = cilclassifier_ckptvit(args, task_id)
                    if ckpt_cilpath_cls.exists():
                        print(f"    Found existing CIL classifier checkpoint for task {task_id} at {ckpt_cilpath_cls}, loading...")
                        clsfr = torch.load(ckpt_cilpath_cls, map_location=args.device)
                    else:

                        clsfr=None

                    (acc, acc_forg), clsfr = vit_evaluate_classifier(
                        mode              = clf_mode,
                        sidenet           = sidevit,
                        global_vit        = global_vit,
                        train_loader      = train_loader,
                        test_loader       = test_loader,
                        num_classes       = len(seen_classes),
                        device            = args.device,
                        epochs_lin        = args.epochs_lin,
                        lr_lin            = args.lr_lin,
                        class_map         = global_to_local,
                        lr_decay          = args.lr_decay_factor,
                        list_test_loaders = list_test_loaders,
                        saved=clsfr
                    )
                    if acc is not None:
                        cil_results[task_id] = acc
                        print(f"    CIL @ task {task_id}: {acc:.2f}%")
                    if acc_forg:
                        for t_idx, fa in enumerate(acc_forg, start=1):
                            print(f"      Forgetting task {t_idx} acc (after task {task_id}): {fa:.2f}%")
                    torch.save(clsfr, ckpt_cilpath_cls)

                if cil_results:
                    print(f"  Avg CIL: {sum(cil_results.values()) / len(cil_results):.2f}%")

if __name__ == "__main__":
    main()

'''

### in case of a TIL with --supervised traing (directly evalalution foromnmn classfier head for cnn, for vit replacing global_dec(images, sidenet) with global_vit(images, sidevit))

if args.til:
    til_results = {}
    for task_id, cfg in tasks.items():
        gckpt = globdec_ckpt(args, task_id)
        if not gckpt.exists():
            continue

        global_dec = build_global_dec(args, num_classes=len(cfg["selected_classes"]))  # [CHANGE] num_classes not None
        global_dec.load_state_dict(torch.load(gckpt, map_location=args.device), strict=False)
        global_dec.eval()
        for p in global_dec.parameters():
            p.requires_grad_(False)

        global_to_local = {
            gid: lidx
            for lidx, gid in enumerate(sorted(cfg["selected_classes"].values()))
        }
        lut = build_lut(global_to_local, args.device)  # from cnn_traintest

        test_loader = make_loader(args, cfg, train=False, batch=args.batch, n_shot=None)

        # [ADD] direct head evaluation — no linear probe
        total = correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                labels = lut[labels]
                _, _, _, _, logits = global_dec(images, sidenet)  # supervised=True forward
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = 100.0 * correct / total
        til_results[task_id] = acc
        print(f"    TIL task {task_id}: {acc:.2f}%")

    if til_results:
        print(f"  Avg TIL: {sum(til_results.values()) / len(til_results):.2f}%")

'''
