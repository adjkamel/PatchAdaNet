import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from cnn_models import PatchNetMulti, GlobalDecoder, PatchNet

USE_AMP = True
torch.backends.cudnn.benchmark = True

#----------------------- Pretrain the per-patch autoencoders (Stage 1).
def train_sidenets(model, dataloader, epochs, device, learn_rate,
                   checkpoints_dir, n_parts, exp, lr_decay):
    model           = model.to(device)
    recon_criterion = nn.MSELoss()
    use_amp         = USE_AMP and device.type == "cuda"

    for b_idx in range(n_parts):
        ckpt_path = Path(checkpoints_dir) / f"cnn_p{n_parts}_side{b_idx}.pth"
        branch    = getattr(model, f"cnn_p{n_parts}_side{b_idx}")

        if ckpt_path.exists():
            branch.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  [Side {b_idx}] Loaded from {ckpt_path}, skipping training.")
            continue

        print(f"  [Side {b_idx}] Training "
              f"{sum(p.numel() for p in branch.parameters()):,} params ...")
        branch.train()
        optimizer = optim.Adam(branch.parameters(), lr=learn_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_decay, patience=5)
        scaler = GradScaler(enabled=use_amp)

        for epoch in range(epochs):
            total_loss = 0.0
            for images, _ in dataloader:
                images  = images.to(device, non_blocking=True)
                patches = model._extract_patches(images)
                patch   = patches[b_idx]
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    recon, _ = branch(patch)
                    loss = recon_criterion(recon, patch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/{epochs} | Recon: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            print(f"    Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(branch.state_dict(), ckpt_path)
        print(f"  [Side {b_idx}] Saved to {ckpt_path}")

#----------------------- Load all pretrained patch-branch checkpoints.
def load_all_sides(model, checkpoints_dir, n_parts, exp, device):
    for b_idx in range(n_parts):
        ckpt_path = Path(checkpoints_dir) / f"cnn_p{n_parts}_side{b_idx}.pth"
        branch    = getattr(model, f"cnn_p{n_parts}_side{b_idx}")
        branch.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded {n_parts} side checkpoints from {checkpoints_dir}")
    return model

#----------------------- Yield the adapter parameters that EWC regularizes.
def _ewc_named_params(global_dec):
    for name, param in global_dec.named_parameters():
        if param.requires_grad:
            if "_frozen_conv_cache" in name:
                continue
            yield name, param

#----------------------- Elastic Weight Consolidation penalty over adapter parameters.
class EWC:
    #----------------------- Initialize the EWC.
    def __init__(self, global_dec, dataloader, sidenet, device,
                 prev_ewc=None, n_batches=50, fisher_scale=1000.0):
        self.device       = device
        self.fisher_scale = fisher_scale

        self._means  = {n: p.data.clone().detach()
                        for n, p in _ewc_named_params(global_dec)}
        self._fisher = {n: torch.zeros_like(p.data)
                        for n, p in _ewc_named_params(global_dec)}

        self._estimate_fisher(global_dec, dataloader, sidenet, n_batches)

        if prev_ewc is not None:
            for name in self._fisher:
                if name in prev_ewc._fisher:
                    self._fisher[name] += prev_ewc._fisher[name]

    #----------------------- Estimate the Fisher information for EWC.
    def _estimate_fisher(self, global_dec, dataloader, sidenet, n_batches):
        global_dec.eval()
        recon_criterion = nn.MSELoss()

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            images = images.to(self.device)
            global_dec.zero_grad(set_to_none=True)
            recon, *_= global_dec(images, sidenet)
            recon_criterion(recon, images).backward()
            for name, param in _ewc_named_params(global_dec):
                if param.grad is not None:
                    self._fisher[name] += param.grad.detach() ** 2

        batches = min(n_batches, len(dataloader))
        for name in self._fisher:
            self._fisher[name] = (self._fisher[name] / max(batches, 1)
                                  * self.fisher_scale).detach()

    #----------------------- Compute the EWC quadratic penalty for the current parameters.
    def penalty(self, global_dec):
        loss = torch.tensor(0.0, device=self.device)
        for name, param in _ewc_named_params(global_dec):
            if name in self._means:
                loss += (self._fisher[name] * (param - self._means[name]) ** 2).sum()
        return loss

#----------------------- Train the GlobalDecoder on one task with GFC and EWC.
def train_global_decoder(global_dec, sidenet, dataloader, epochs, device,
                         learn_rate, ewc=None, ewc_lambda=5000.0,
                         lr_decay=0.05, lambda_proto=0.1, task_id=1, use_proto=False,class_map=None,supervised=False,train_sidenet=False):

    sidenet = sidenet.to(device)
    if not train_sidenet:
        sidenet.eval()
        for p in sidenet.parameters():
            p.requires_grad_(False)
    else:
        sidenet.train()

        for p in sidenet.parameters():
            p.requires_grad_(True)

    global_dec = global_dec.to(device)
    global_dec.invalidate_frozen_cache()
    global_dec.cache_frozen_convs(sidenet)
    print("  [GlobalDec] Frozen-conv cache built.")

    recon_criterion  = nn.MSELoss()
    class_criterion  = nn.CrossEntropyLoss()
    trainable_params = [p for p in global_dec.parameters() if p.requires_grad]
    if train_sidenet:
        trainable_params += [p for p in sidenet.parameters() if p.requires_grad]
    print(f"  [GlobalDec] Trainable: {sum(p.numel() for p in trainable_params):,} params")
    print(f"  [GlobalDec] lambda_proto={lambda_proto} | task_id={task_id} | "
          f"EWC={'enabled' if ewc else 'disabled'}")

    optimizer = optim.Adam(trainable_params, lr=learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay, patience=5)
    use_amp = USE_AMP and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    last_global_layers = None
    last_frozen_layers = None

    lut = build_lut(class_map, device) if (supervised and class_map) else None

    n_batches_total = len(dataloader)
    for epoch in range(epochs):
        global_dec.train()
        total_recon = total_proto = total_ewc = total_class = 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if lut is not None:
                labels = lut[labels]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                if supervised:
                    recon, bottleneck, global_layers, frozen_layers, out_class = global_dec(images, sidenet)
                else:
                    recon, bottleneck, global_layers, frozen_layers = global_dec(images, sidenet)

                recon_loss = recon_criterion(recon, images)

                if supervised:

                    class_loss = class_criterion(out_class, labels)

                if use_proto:
                    proto_loss = global_dec.prototype.alignment_loss(global_layers)

                ewc_loss = (ewc_lambda * ewc.penalty(global_dec)
                            if ewc is not None
                            else torch.tensor(0.0, device=device))

                if use_proto:
                    if supervised:
                        loss = recon_loss + lambda_proto * proto_loss + ewc_loss+ class_loss
                    else:
                        loss = recon_loss + lambda_proto * proto_loss + ewc_loss

                else:
                    if supervised:
                        loss = recon_loss  + ewc_loss + class_loss
                    else:
                        loss = recon_loss  + ewc_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_recon += recon_loss.item()
            if supervised:
                total_class += class_loss.item()
            if use_proto:
                total_proto += proto_loss.item()
            total_ewc   += ewc_loss.item()

            last_global_layers = [gl.detach() for gl in global_layers]
            last_frozen_layers = [fl.detach() for fl in frozen_layers]

        n       = len(dataloader)
        ewc_str = f" | EWC: {total_ewc/n:.4f}" if ewc is not None else ""
        proto_str = f" | GFC: {total_proto/n:.4f}" if use_proto else ""
        print(f"  [CNN-ImageAE] Epoch {epoch+1}/{epochs}"
              f" | Recon: {total_recon/n:.4f}{proto_str}{ewc_str}"
              f" | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if use_proto:
            scheduler.step((total_recon + total_proto + total_ewc) / n)
        else:
            scheduler.step((total_recon + total_ewc) / n)

    if use_proto:
        if last_frozen_layers is not None and last_global_layers is not None:
            if task_id == 1:
                global_dec.prototype.initialise(last_frozen_layers)
            else:
                global_dec.prototype.update(last_global_layers)

        tensor_list_proto = [global_dec.prototype.maps[0], global_dec.prototype.maps[1], global_dec.prototype.maps[2]]

    print("  [GlobalDec] Computing Fisher information for EWC ...")
    new_ewc = EWC(global_dec, dataloader, sidenet, device, prev_ewc=ewc)

    if not use_proto:
       return global_dec, new_ewc

    return global_dec, new_ewc, tensor_list_proto

#----------------------- Extract classifier features for a data loader.
def extract_global_features(global_dec, sidenet, images):
    global_dec.eval()
    sidenet.eval()
    with torch.no_grad():
        return global_dec.get_classifier_features(images, sidenet)

#----------------------- Build the global-to-local class-index lookup table.
def build_lut(class_map, device):
    max_key = max(class_map.keys()) + 1
    lut = torch.zeros(max_key, dtype=torch.long, device=device)
    for k, v in class_map.items():
        lut[k] = v
    return lut

#----------------------- Train the linear classifier on frozen features.
def train_linear_classifier(feat_fn, loader, num_classes, device,
                             epochs, lr, class_map, lr_decay):
    x, _       = next(iter(loader))
    feat_dim   = feat_fn(x.to(device)).shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer  = optim.Adam(classifier.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay, patience=5)
    criterion  = nn.CrossEntropyLoss()
    lut        = build_lut(class_map, device) if class_map else None
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if lut is not None:
                labels = lut[labels]
            loss = criterion(classifier(feat_fn(images)), labels)
            total_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(loader)
        print(f"    Linear Classifier Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        print(f"    Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    return classifier

#----------------------- Evaluate the linear classifier and return accuracy.
def eval_linear_classifier(feat_fn, classifier, loader, device, class_map, list_test_loaders=None):
    classifier.eval()
    lut = build_lut(class_map, device) if class_map else None

    acc = None
    if loader is not None:
        total = correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if lut is not None:
                    labels = lut[labels]
                preds   = classifier(feat_fn(images)).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = 100.0 * correct / total

    acc_forg = []
    if list_test_loaders:
        for ts_loader, task_class_map in list_test_loaders:
            task_lut = build_lut(task_class_map, device) if task_class_map else None
            total = correct = 0
            with torch.no_grad():
                for images, labels in ts_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    if task_lut is not None:
                        labels = task_lut[labels]
                    preds = classifier(feat_fn(images)).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            acc_forg.append(100.0 * correct / total)

    return acc, acc_forg

#----------------------- Run CIL/TIL evaluation across all tasks for the CNN model.
def cnn_evaluate_classifier(mode, sidenet, global_dec, train_loader, test_loader,
                             num_classes, device, epochs_lin, lr_lin,
                             class_map, lr_decay, list_test_loaders=None, saved=None):
    sidenet.eval()
    for p in sidenet.parameters():
        p.requires_grad_(False)
    global_dec.eval()
    for p in global_dec.parameters():
        p.requires_grad_(False)

    #----------------------- Feature-extraction closure passed to the evaluator.
    def feat_fn(imgs):
        return extract_global_features(global_dec, sidenet, imgs)

    if saved != None:
      clf=saved
    else:
        clf = train_linear_classifier(feat_fn, train_loader, num_classes,
                                      device, epochs_lin, lr_lin, class_map, lr_decay)

    return eval_linear_classifier(feat_fn, clf, test_loader, device, class_map,list_test_loaders), clf
