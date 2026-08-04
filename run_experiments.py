
import argparse
import json
import os
import random
import sys
import numpy as np
from pathlib import Path

SEED_RUNS = {
    1: 15,
    2: 15,
    3: 5,
    4: 5,
    5: 5,
}

CLASS_ORDER_RUNS = {
    1: 5,
    2: 5,
    3: 5,
    5: 5,
}

CLASSES = {
    1: ["airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"],

    2: [
        "beaver","dolphin","otter","seal","whale",
        "aquarium_fish","flatfish","ray","shark","trout",
        "orchid","poppy","rose","sunflower","tulip",
        "bottle","bowl","can","cup","plate",
        "apple","mushroom","orange","pear","sweet_pepper",
        "clock","keyboard","lamp","telephone","television",
        "bed","chair","couch","table","wardrobe",
        "bee","beetle","butterfly","caterpillar","cockroach",
        "bear","leopard","lion","tiger","wolf",
        "bridge","castle","house","road","skyscraper",
        "cloud","forest","mountain","plain","sea",
        "camel","cattle","chimpanzee","elephant","kangaroo",
        "fox","porcupine","possum","raccoon","skunk",
        "crab","lobster","snail","spider","worm",
        "baby","boy","girl","man","woman",
        "crocodile","dinosaur","lizard","snake","turtle",
        "hamster","mouse","rabbit","shrew","squirrel",
        "maple_tree","oak_tree","palm_tree","pine_tree","willow_tree",
        "bicycle","bus","motorcycle","pickup_truck","train",
        "lawn_mower","rocket","streetcar","tank","tractor",
    ],

    3: [str(i) for i in range(100)],

    5: (
        [str(i) for i in range(10)] +
        [f"n{i}" for i in range(10)] +
        list("ABCDEFGHIJ") +
        ["Tshirt_top","Trouser","Pullover","Dress","Coat",
         "Sandal","Shirt","Sneaker","Bag","Ankle_boot"] +
        ["airplane","automobile","bird","cat","deer",
         "dog","frog","horse","ship","truck"]
    ),
}

CLASSES_PER_TASK = {1: 2, 2: 5, 3: 5, 4: 10, 5: 10}

#----------------------- Set all RNG seeds for Stage-2 training reproducibility.
def apply_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass

#----------------------- Return a task dict with randomly shuffled class-to-task assignment.
def shuffled_tasks(experiment: int, run_idx: int) -> dict:
    import importlib
    main_mod = importlib.import_module("main")

    base = main_mod.get_tasks(experiment, img_size=64, data_root="./")
    n_tasks = len(base)

    classes     = list(CLASSES[experiment])
    cpt         = CLASSES_PER_TASK[experiment]
    rng         = random.Random(run_idx * 1009 + 17)
    rng.shuffle(classes)

    dc  = {n: i for i, n in enumerate(CLASSES[experiment])}
    shuffled = {}
    for t in range(n_tasks):
        task_classes = classes[t * cpt : (t + 1) * cpt]
        selected     = {name: dc[name] for name in task_classes}
        cfg          = dict(base[t + 1])
        cfg["selected_classes"] = selected
        shuffled[t + 1] = cfg
    return shuffled

#----------------------- Write a run's CIL/TIL results to a JSON file.
def save_results(results_dir: Path, tag: str, cil_results: dict, til_results: dict):
    out = {"cil": cil_results, "til": til_results}
    path = results_dir / f"{tag}_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {path}")
    return out

#----------------------- Print mean +/- std of average CIL and TIL across runs.
def summarise(all_cil: list, all_til: list, label: str):
    #----------------------- Return the mean of a dict of per-task accuracies.
    def avg_of_dict(d):
        vals = list(d.values())
        return sum(vals) / len(vals) if vals else None

    cil_avgs = [avg_of_dict(r) for r in all_cil if r]
    til_avgs = [avg_of_dict(r) for r in all_til if r]

    print(f"\n{'='*70}")
    print(f"  [{label}]")
    if cil_avgs:
        arr = np.array(cil_avgs)
        print(f"  Avg CIL across runs: {arr.mean():.2f} ± {arr.std():.2f}%")
        print(f"  Per-run CIL avgs:    {[f'{v:.2f}' for v in cil_avgs]}")
    if til_avgs:
        arr = np.array(til_avgs)
        print(f"  Avg TIL across runs: {arr.mean():.2f} ± {arr.std():.2f}%")
        print(f"  Per-run TIL avgs:    {[f'{v:.2f}' for v in til_avgs]}")
    print(f"{'='*70}\n")

#----------------------- Execute one Stage-2 training and evaluation run in-process.
def run_one_inprocess(args, tasks_override=None):
    import torch
    import importlib

    main_mod   = importlib.import_module("main")
    _cnn_mod_name = "cnn_traintest_fast" if getattr(args, "fast_cnn", False) else "cnn_traintest"
    cnn_train  = importlib.import_module(_cnn_mod_name)
    vit_train  = importlib.import_module("vit_traintest")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)

    tasks = tasks_override if tasks_override is not None else\
            main_mod.get_tasks(args.experiment, args.img_size, args.data_root)
    num_tasks = len(tasks)

    cil_results = {}
    til_results = {}

    if args.model == "cnn":
        from cnn_models import PatchNetMulti, GlobalDecoder

        sidenet = main_mod.build_sidenet(args, grayscale=False)
        sidenet = cnn_train.load_all_sides(
            sidenet, args.side_checkpoints, args.n_parts, args.experiment, device)

        if args.learn_global:
            ewc = None
            for task_id in sorted(tasks.keys()):
                print(f"\n  [CNN] Task {task_id}/{num_tasks}")
                train_loader = main_mod.make_loader(
                    args, tasks[task_id], train=True, batch=args.batch,
                    n_shot=getattr(args, "n_shot", None))

                global_dec = main_mod.build_global_dec(
                    args, num_classes=len(tasks[task_id]["selected_classes"]))

                prev = main_mod.globdec_ckpt(args, task_id - 1)
                if prev.exists():
                    sd = torch.load(prev, map_location=device)
                    global_dec.load_state_dict(sd, strict=False)
                    if any("prototype.maps" in k for k in sd):
                        global_dec.prototype.is_initialised.fill_(True)
                    global_dec.invalidate_frozen_cache()

                local_class_map = {
                    gid: lidx
                    for lidx, gid in enumerate(
                        sorted(tasks[task_id]["selected_classes"].values()))
                }

                if args.prototype:
                    global_dec, ewc, tensor_proto = cnn_train.train_global_decoder(
                        global_dec=global_dec, sidenet=sidenet,
                        dataloader=train_loader, epochs=args.epochs_glob,
                        device=device, learn_rate=args.lr_glob, ewc=ewc,
                        ewc_lambda=args.ewc_lambda, lr_decay=args.lr_decay_factor,
                        lambda_proto=args.proto_alpha, task_id=task_id,
                        use_proto=args.prototype, class_map=local_class_map,
                        supervised=args.supervised,
                        train_sidenet=args.train_sidenet_global)
                    for pi, tp in enumerate(tensor_proto, 1):
                        torch.save(tp, main_mod.cnnproto_ckpt(args, f"p{pi}", task_id))
                else:
                    global_dec, ewc = cnn_train.train_global_decoder(
                        global_dec=global_dec, sidenet=sidenet,
                        dataloader=train_loader, epochs=args.epochs_glob,
                        device=device, learn_rate=args.lr_glob, ewc=ewc,
                        ewc_lambda=args.ewc_lambda, lr_decay=args.lr_decay_factor,
                        lambda_proto=args.proto_alpha, task_id=task_id,
                        use_proto=args.prototype, class_map=local_class_map,
                        supervised=args.supervised,
                        train_sidenet=args.train_sidenet_global)

                torch.save(global_dec.state_dict(),
                           main_mod.globdec_ckpt(args, task_id))

        if args.eval:
            clf_mode = "linear_global"
            if args.til:
                for task_id, cfg in tasks.items():
                    gckpt = main_mod.globdec_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    global_dec = main_mod.build_global_dec(args, num_classes=None)
                    global_dec.load_state_dict(
                        torch.load(gckpt, map_location=device), strict=False)
                    num_classes     = len(cfg["selected_classes"])
                    global_to_local = {
                        gid: lidx for lidx, gid in
                        enumerate(sorted(cfg["selected_classes"].values()))
                    }
                    tr = main_mod.make_loader(args, cfg, train=True,  batch=args.batch)
                    te = main_mod.make_loader(args, cfg, train=False, batch=args.batch)
                    (acc, _), _ = cnn_train.cnn_evaluate_classifier(
                        clf_mode, sidenet, global_dec, tr, te, num_classes,
                        device, args.epochs_lin, args.lr_lin,
                        global_to_local, args.lr_decay_factor)
                    til_results[task_id] = acc
                    print(f"    TIL task {task_id}: {acc:.2f}%")
                if til_results:
                    print(f"  Avg TIL: {sum(til_results.values())/len(til_results):.2f}%")

            if args.cil:
                seen_classes = {}
                for task_id in sorted(tasks.keys()):
                    gckpt = main_mod.globdec_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    seen_classes.update(tasks[task_id]["selected_classes"])
                    global_dec = main_mod.build_global_dec(args, num_classes=None)
                    global_dec.load_state_dict(
                        torch.load(gckpt, map_location=device), strict=False)
                    global_to_local = {
                        gid: lidx for lidx, gid in
                        enumerate(sorted(seen_classes.values()))
                    }
                    base_cfg = dict(tasks[1])
                    base_cfg["selected_classes"] = dict(seen_classes)
                    tr = main_mod.make_loader(args, base_cfg, train=True,  batch=args.batch)
                    te = main_mod.make_loader(args, base_cfg, train=False, batch=args.batch)
                    (acc, _), _ = cnn_train.cnn_evaluate_classifier(
                        clf_mode, sidenet, global_dec, tr, te, len(seen_classes),
                        device, args.epochs_lin, args.lr_lin,
                        global_to_local, args.lr_decay_factor)
                    if acc is not None:
                        cil_results[task_id] = acc
                        print(f"    CIL @ task {task_id}: {acc:.2f}%")
                if cil_results:
                    print(f"  Avg CIL: {sum(cil_results.values())/len(cil_results):.2f}%")

    elif args.model == "vit":
        sidevit = main_mod.build_sidevit(args, grayscale=False)
        sidevit = vit_train.load_all_sidevits(
            sidevit, args.side_checkpoints, args.n_parts, args.experiment, device)

        if args.learn_global:
            ewc = None
            for task_id in sorted(tasks.keys()):
                print(f"\n  [ViT] Task {task_id}/{num_tasks}")
                train_loader = main_mod.make_loader(
                    args, tasks[task_id], train=True, batch=args.batch,
                    n_shot=getattr(args, "n_shot", None))

                global_vit = main_mod.build_global_vit(
                    args, num_classes=len(tasks[task_id]["selected_classes"]))

                prev = main_mod.globvit_ckpt(args, task_id - 1)
                if prev.exists():
                    global_vit.load_state_dict(
                        torch.load(prev, map_location=device))
                    global_vit.invalidate_frozen_cache()

                local_class_map = {
                    gid: lidx
                    for lidx, gid in enumerate(
                        sorted(tasks[task_id]["selected_classes"].values()))
                }

                if args.prototype:
                    global_vit, ewc, tensor_proto = vit_train.train_global_vit(
                        global_vit=global_vit, sidenet=sidevit,
                        dataloader=train_loader, epochs=args.epochs_glob,
                        device=device, learn_rate=args.lr_glob, ewc=ewc,
                        ewc_lambda=args.ewc_lambda, lr_decay=args.lr_decay_factor,
                        lambda_proto=args.proto_alpha, task_id=task_id,
                        use_proto=args.prototype, class_map=local_class_map,
                        supervised=args.supervised,
                        train_sidenet=args.train_sidenet_global)
                    for pi, tp in enumerate(tensor_proto, 1):
                        torch.save(tp, main_mod.cnnproto_ckpt(args, f"p{pi}", task_id))
                else:
                    global_vit, ewc = vit_train.train_global_vit(
                        global_vit=global_vit, sidenet=sidevit,
                        dataloader=train_loader, epochs=args.epochs_glob,
                        device=device, learn_rate=args.lr_glob, ewc=ewc,
                        ewc_lambda=args.ewc_lambda, lr_decay=args.lr_decay_factor,
                        lambda_proto=args.proto_alpha, task_id=task_id,
                        use_proto=args.prototype, class_map=local_class_map,
                        supervised=args.supervised,
                        train_sidenet=args.train_sidenet_global)

                torch.save(global_vit.state_dict(),
                           main_mod.globvit_ckpt(args, task_id))

        if args.eval:
            clf_mode = "linear_global"
            if args.til:
                for task_id, cfg in tasks.items():
                    gckpt = main_mod.globvit_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    global_vit = main_mod.build_global_vit(args, num_classes=None)
                    global_vit.load_state_dict(
                        torch.load(gckpt, map_location=device), strict=False)
                    num_classes     = len(cfg["selected_classes"])
                    global_to_local = {
                        gid: lidx for lidx, gid in
                        enumerate(sorted(cfg["selected_classes"].values()))
                    }
                    tr = main_mod.make_loader(args, cfg, train=True,  batch=args.batch)
                    te = main_mod.make_loader(args, cfg, train=False, batch=args.batch)
                    (acc, _), _ = vit_train.vit_evaluate_classifier(
                        clf_mode, sidevit, global_vit, tr, te, num_classes,
                        device, args.epochs_lin, args.lr_lin,
                        global_to_local, args.lr_decay_factor)
                    til_results[task_id] = acc
                    print(f"    TIL task {task_id}: {acc:.2f}%")
                if til_results:
                    print(f"  Avg TIL: {sum(til_results.values())/len(til_results):.2f}%")

            if args.cil:
                seen_classes = {}
                for task_id in sorted(tasks.keys()):
                    gckpt = main_mod.globvit_ckpt(args, task_id)
                    if not gckpt.exists():
                        continue
                    seen_classes.update(tasks[task_id]["selected_classes"])
                    global_vit = main_mod.build_global_vit(args, num_classes=None)
                    global_vit.load_state_dict(
                        torch.load(gckpt, map_location=device), strict=False)
                    global_to_local = {
                        gid: lidx for lidx, gid in
                        enumerate(sorted(seen_classes.values()))
                    }
                    base_cfg = dict(tasks[1])
                    base_cfg["selected_classes"] = dict(seen_classes)
                    tr = main_mod.make_loader(args, base_cfg, train=True,  batch=args.batch)
                    te = main_mod.make_loader(args, base_cfg, train=False, batch=args.batch)
                    (acc, _), _ = vit_train.vit_evaluate_classifier(
                        clf_mode, sidevit, global_vit, tr, te, len(seen_classes),
                        device, args.epochs_lin, args.lr_lin,
                        global_to_local, args.lr_decay_factor)
                    if acc is not None:
                        cil_results[task_id] = acc
                        print(f"    CIL @ task {task_id}: {acc:.2f}%")
                if cil_results:
                    print(f"  Avg CIL: {sum(cil_results.values())/len(cil_results):.2f}%")

    return cil_results, til_results

#----------------------- Parse run-harness args and pass the rest through to args.
def parse_run_args():
    import args as ra
    base_parser = ra.build_parser()

    if not any(a.dest == 'resume_from' for a in base_parser._actions):
        base_parser.add_argument(
            "--resume_from", type=int, default=0,
            help="Resume multi-run experiment from this run index (0-based). "
                 "Runs before this index are skipped entirely. "
                 "For seed_fixed_order: --resume_from 3 skips seeds 0,1,2 and starts from seed 3. "
                 "For class_order: --resume_from 2 skips class orders 0,1 and starts from order 2.")
    return base_parser.parse_args()

#----------------------- Entry point: dispatch single, seed, or class-order run modes.
def main():
    args        = parse_run_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp         = args.experiment
    run_mode    = args.run_mode

    print(f"\nPatchAdaNet multi-run harness")
    print(f"  model={args.model} | experiment={exp} | run_mode={run_mode}")
    print(f"  results_dir={results_dir}\n")

    if run_mode == "single":
        cil, til = run_one_inprocess(args)
        save_results(results_dir, f"exp{exp}_{args.model}_seed{seed}", cil, til)
        return

    if run_mode == "seed_fixed_order":
        n_runs = SEED_RUNS.get(exp)
        if n_runs is None:
            raise ValueError(
                f"seed_fixed_order not supported for experiment {exp}. "
                f"Valid: {sorted(SEED_RUNS)}")

        all_cil, all_til = [], []
        base_ckpt   = args.checkpoints
        resume_from = getattr(args, 'resume_from', 0)

        if resume_from > 0:
            print(f"\n  [Resume] Skipping runs 0..{resume_from-1}, starting from run {resume_from}.")
            print(f"  [Resume] Loading saved results for skipped runs from {results_dir} ...")
            for skip_idx in range(resume_from):
                tag  = f"exp{exp}_{args.model}_seed{skip_idx}"
                path = results_dir / f"{tag}_results.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    all_cil.append({int(k): v for k, v in data.get("cil", {}).items()})
                    all_til.append({int(k): v for k, v in data.get("til", {}).items()})
                    print(f"    Loaded {path.name}")
                else:
                    print(f"    WARNING: {path} not found — inserting empty result for run {skip_idx}")
                    all_cil.append({})
                    all_til.append({})

        for run_idx in range(resume_from, n_runs):
            seed = run_idx
            print(f"\n{'#'*70}")
            print(f"# RUN {run_idx+1}/{n_runs} | seed={seed}")
            print(f"{'#'*70}")

            apply_seed(seed)

            args.checkpoints = str(Path(base_ckpt) / f"seed_{seed}")
            args.side_checkpoints = base_ckpt
            Path(args.checkpoints).mkdir(parents=True, exist_ok=True)

            cil, til = run_one_inprocess(args)
            all_cil.append(cil)
            all_til.append(til)
            save_results(results_dir, f"exp{exp}_{args.model}_seed{seed}", cil, til)

        args.checkpoints = base_ckpt
        summarise(all_cil, all_til, f"exp{exp} seed_fixed_order ({n_runs} runs)")
        with open(results_dir / f"exp{exp}_{args.model}_seed_fixed_order_summary.json", "w") as f:
            json.dump({"mode": "seed_fixed_order", "n_runs": n_runs,
                       "cil_runs": all_cil, "til_runs": all_til}, f, indent=2)
        return

    if run_mode == "class_order":
        n_runs = CLASS_ORDER_RUNS.get(exp)
        if n_runs is None:
            raise ValueError(
                f"class_order not supported for experiment {exp}. "
                f"Valid: {sorted(CLASS_ORDER_RUNS)}")

        all_cil, all_til, all_orders = [], [], []
        base_ckpt   = args.checkpoints
        resume_from = getattr(args, 'resume_from', 0)

        if resume_from > 0:
            print(f"\n  [Resume] Skipping class orders 0..{resume_from-1}, starting from order {resume_from}.")
            print(f"  [Resume] Loading saved results for skipped runs from {results_dir} ...")
            for skip_idx in range(resume_from):
                tag  = f"exp{exp}_{args.model}_classorder_{skip_idx}"
                path = results_dir / f"{tag}_results.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    all_cil.append({int(k): v for k, v in data.get("cil", {}).items()})
                    all_til.append({int(k): v for k, v in data.get("til", {}).items()})
                    print(f"    Loaded {path.name}")
                else:
                    print(f"    WARNING: {path} not found — inserting empty result for order {skip_idx}")
                    all_cil.append({})
                    all_til.append({})
                all_orders.append(skip_idx)

        for run_idx in range(resume_from, n_runs):
            print(f"\n{'#'*70}")
            print(f"# RUN {run_idx+1}/{n_runs} | class_order run {run_idx}")
            print(f"{'#'*70}")

            tasks_override = shuffled_tasks(exp, run_idx)

            class_order = {t: list(tasks_override[t]["selected_classes"].items())
                           for t in sorted(tasks_override)}
            for t, cls in class_order.items():
                print(f"  Task {t}: {cls}")

            args.checkpoints = str(Path(base_ckpt) / f"classorder_{run_idx}")
            args.side_checkpoints = base_ckpt
            Path(args.checkpoints).mkdir(parents=True, exist_ok=True)

            cil, til = run_one_inprocess(args, tasks_override=tasks_override)
            all_cil.append(cil)
            all_til.append(til)
            all_orders.append(class_order)

            save_results(results_dir, f"exp{exp}_{args.model}_classorder_{run_idx}", cil, til)
            with open(results_dir / f"exp{exp}_{args.model}_classorder_{run_idx}_order.json", "w") as f:
                json.dump(class_order, f, indent=2)

        args.checkpoints = base_ckpt
        summarise(all_cil, all_til, f"exp{exp} class_order ({n_runs} runs)")
        with open(results_dir / f"exp{exp}_{args.model}_class_order_summary.json", "w") as f:
            json.dump({"mode": "class_order", "n_runs": n_runs,
                       "class_orders": all_orders,
                       "cil_runs": all_cil, "til_runs": all_til}, f, indent=2)
        return

if __name__ == "__main__":
    main()