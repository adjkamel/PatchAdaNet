import os
import torch
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

#----------------------- Return (mean, std) normalization tuples matched to each dataset.
def _get_normalization(dataset_name, grayscale):
    name = dataset_name.lower()

    if "mnist" in name:

        return (0.1307,), (0.3081,)

    if "fashion" in name:

        return (0.2860,), (0.3530,)

    if "cifar100" in name:
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    if "cifar10" in name:

        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    if "tinyimagenet" in name or "tiny_imagenet" in name or "tinyimagenet" in name:
        return (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)

    if "miniimagenet" in name or "mini_imagenet" in name:
        return (0.4733, 0.4490, 0.4029), (0.2796, 0.2722, 0.2863)

    if "imagenet" in name:

        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if "five_data" in name or "five-data" in name:

        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if "svhn" in name:
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

    if "stl" in name:
        return (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)

    if grayscale:
        print(f"[WARNING] Unknown dataset '{dataset_name}' – using MNIST normalization.")
        return (0.1307,), (0.3081,)
    else:
        print(f"[WARNING] Unknown dataset '{dataset_name}' – using 0.5/0.5 normalization.")
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

#----------------------- General-purpose image-folder dataset for continual learning tasks.
class ContinualDataset(Dataset):

    #----------------------- Initialize the ContinualDataset.
    def __init__(self, dataset_name, root="./", train=True,
                 selected_classes=None, img_size=32, grayscale=True,
                 transform=None, seed=42, permute=False, task_num=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.train = train
        self.img_size = img_size
        self.grayscale = grayscale
        self.permute = permute
        self.seed = seed
        self.selected_classes_arg = selected_classes

        self.class_to_id = None
        self.selected_class_list = None

        if isinstance(selected_classes, dict):
            self.class_to_id = {str(k): int(v) for k, v in selected_classes.items()}
        elif isinstance(selected_classes, list):
            self.selected_class_list = [str(c) for c in selected_classes]

        if self.permute:
            idx = list(range(img_size * img_size))
            self.permute_idx = torch.tensor(
                shuffle(idx, random_state=seed * 100 + task_num),
                dtype=torch.long
            )
        else:
            self.permute_idx = None

        if transform is None:
            mean, std = _get_normalization(dataset_name, grayscale)
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1) if grayscale
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transform

        self.image_paths, self.labels = [], []
        self.root_dir = Path(root) / dataset_name / ("train" if train else "test")
        print('------------------------------------',
              f"Loading dataset from: {self.root_dir}")

        self._load_from_folders()
        self._filter_classes()

        print(f"[{dataset_name.upper()} - {'Train' if train else 'Test'}] "
              f"Loaded {len(self.image_paths)} images. Permutation: {self.permute}")

    #----------------------- Load image paths and labels from the class subfolders.
    def _load_from_folders(self):
        for class_folder in sorted(os.listdir(self.root_dir)):
            class_path = self.root_dir / class_folder
            if not class_path.is_dir():
                continue
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPEG"]:
                for img_path in class_path.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(str(class_folder))

    #----------------------- Keep only the classes selected for the current task.
    def _filter_classes(self):
        if self.class_to_id is not None:
            valid   = set(self.class_to_id.keys())
            indices = [i for i, lbl in enumerate(self.labels) if str(lbl) in valid]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels      = [str(self.labels[i])  for i in indices]
            return

        if self.selected_class_list is not None:
            valid   = set(self.selected_class_list)
            indices = [i for i, lbl in enumerate(self.labels) if str(lbl) in valid]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels      = [str(self.labels[i])  for i in indices]

    #----------------------- Return the number of samples in the dataset.
    def __len__(self):
        return len(self.image_paths)

    #----------------------- Return the transformed image and label at an index.
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        img = Image.open(img_path).convert('L' if self.grayscale else 'RGB')
        img = self.transform(img)

        if self.permute and self.permute_idx is not None:
            c, h, w = img.shape
            flat    = img.view(-1)
            img     = flat[self.permute_idx].view(c, h, w)

        if self.class_to_id is not None:
            label = self.class_to_id[str(label)]
        else:
            try:
                label = int(label)
            except Exception:
                pass

        return img, label

#----------------------- Choose a DataLoader worker count from available CPU cores.
def _optimal_workers():
    cpu_count = os.cpu_count() or 1
    return min(8, max(4, cpu_count // 2))

#----------------------- Build train/test DataLoaders for a task configuration.
def create_continual_loaders(cfg,  sub_perc=0.25, batch_size=64, shuffle=True,
                             crop_mode="full", permute=False, n_shot=None):
    ds = ContinualDataset(
        dataset_name    = cfg["name"],
        root            = cfg.get("root", "./"),
        train           = cfg.get("train", True),
        selected_classes= cfg.get("selected_classes", None),
        img_size        = cfg.get("img_size", 32),
        grayscale       = cfg.get("grayscale", True),
        permute         = cfg.get("permute", False),
        task_num        = cfg.get("pidx", 0),
        seed            = cfg.get("seed", 42),
    )

    num_workers = _optimal_workers()

    if n_shot is not None:
        subset = get_fewshot_from_dataset(ds, n_shot=n_shot,seed=42)
        new_ds=subset

    else:
        new_ds=ds

    return DataLoader(
        new_ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        prefetch_factor = 2,
        persistent_workers = (num_workers > 0),
        drop_last   = False,
    )

#----------------------- Create a few-shot subset from a ContinualDataset instance.
def get_fewshot_from_dataset(dataset, n_shot=5, seed=42):
    np.random.seed(seed)

    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)

    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)

    class_to_indices = {}
    for class_id in unique_classes:
        class_to_indices[class_id] = np.where(all_labels == class_id)[0]

    selected_indices = []
    for class_id, indices in class_to_indices.items():
        if len(indices) < n_shot:
            print(f"[WARNING] Class {class_id} has only {len(indices)} samples, using all of them")
            chosen = indices
        else:
            chosen = np.random.choice(indices, size=n_shot, replace=False)
        selected_indices.extend(chosen)

    return Subset(dataset, selected_indices)
