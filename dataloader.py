"""
Dataset classes for BCSS-WSSS
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils import generate_csv_from_filenames, get_conch_norm_stats

# Global CONCH normalization stats (will be updated after loading CONCH)
CONCH_MEAN = [0.48145466, 0.4578275, 0.40821073]
CONCH_STD = [0.26862954, 0.26130258, 0.27577711]


def update_conch_norm(mean, std):
    """Update global CONCH normalization stats"""
    global CONCH_MEAN, CONCH_STD
    CONCH_MEAN = mean
    CONCH_STD = std


class WSSS_Dataset(Dataset):
    """Dataset for weakly-supervised classification training"""
    
    def __init__(self, root_dir, label_file, target_size=448, mean=None, std=None):
        """
        Args:
            root_dir: Directory containing training images
            label_file: Path to CSV file with labels
            target_size: Target image size
            mean: Normalization mean (defaults to CONCH_MEAN)
            std: Normalization std (defaults to CONCH_STD)
        """
        self.root_dir = root_dir
        if (not os.path.exists(label_file)) and os.path.isdir(root_dir):
            print(f"Label CSV not found: {label_file}. Auto-generating from filenames...")
            generate_csv_from_filenames(root_dir, label_file)

        self.df = pd.read_csv(label_file)
        self.target_size = int(target_size)

        mean = CONCH_MEAN if mean is None else mean
        std = CONCH_STD if std is None else std

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(5):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.root_dir, row["filename"])
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                labels = torch.tensor([
                    row["Tumor"],
                    row["Stroma"],
                    row["Lymphocytic infiltrate"],
                    row["Necrosis"],
                ], dtype=torch.float32)
                return image, labels
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                idx = (idx + 1) % len(self.df)
        return torch.zeros(3, self.target_size, self.target_size), torch.zeros(4)


class BCSSValDataset(Dataset):
    """
    Dataset for validation/test with pixel-level annotations
    
    Returns:
        img_tensor: float [3,S,S]
        img_rgb:    uint8 [S,S,3]  (for tissue mask/vis)
        gt:         long  [S,S]    (0..4 with 0=BG)
    """
    
    def __init__(self, img_dir, mask_dir, target_size=448, mean=None, std=None):
        """
        Args:
            img_dir: Directory containing images
            mask_dir: Directory containing masks
            target_size: Target image size
            mean: Normalization mean (defaults to CONCH_MEAN)
            std: Normalization std (defaults to CONCH_STD)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = int(target_size)

        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) \
            if os.path.exists(img_dir) else []

        mean = CONCH_MEAN if mean is None else mean
        std = CONCH_STD if std is None else std

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_p = os.path.join(self.img_dir, fname)
        msk_p = os.path.join(self.mask_dir, fname)

        img_pil = Image.open(img_p).convert("RGB").resize((self.target_size, self.target_size), resample=Image.BICUBIC)
        img_rgb = np.array(img_pil).astype(np.uint8)
        img_tensor = self.transform(img_pil)

        # GT: raw 0=T,1=S,2=L,3=N,4=BG -> remap to 0..4 with 0=BG
        if os.path.exists(msk_p):
            gt_pil = Image.open(msk_p).resize((self.target_size, self.target_size), resample=Image.NEAREST)
            gt_raw = np.array(gt_pil).astype(np.int64)

            gt = np.zeros_like(gt_raw)
            gt[gt_raw == 4] = 0
            gt[gt_raw == 0] = 1
            gt[gt_raw == 1] = 2
            gt[gt_raw == 2] = 3
            gt[gt_raw == 3] = 4
        else:
            gt = np.zeros((self.target_size, self.target_size), dtype=np.int64)

        return {
            "img_tensor": img_tensor,
            "img_rgb": torch.from_numpy(img_rgb).to(torch.uint8),
            "gt": torch.from_numpy(gt).to(torch.long),
            "fname": fname
        }

