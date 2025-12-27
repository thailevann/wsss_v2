"""
Utility functions for BCSS-WSSS training
"""
import os
import random
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prototypes_cluster_bank(path, device, use_all=True):
    """
    Load prototype features from file (supports .pt and .pkl)
    
    Args:
        path: Path to prototype file
        device: Device to load on
        use_all: If True, return all features; if False, return per-class mean
        
    Returns:
        torch.Tensor: Prototype features [N, D] or [4, D]
    """
    assert os.path.exists(path), f"Prototype file not found: {path}"
    data = None
    try:
        data = torch.load(path, map_location=device)
    except Exception as e_torch:
        try:
            with open(path, "rb") as f:
                data = pkl.load(f)
        except Exception as e_pkl:
            raise RuntimeError(
                f"Cannot load prototype file.\n"
                f"- torch.load error: {e_torch}\n"
                f"- pickle.load error: {e_pkl}\n"
            )

    if not isinstance(data, dict):
        raise TypeError(f"Prototype content must be dict, got: {type(data)}")
    if "features" not in data or "cumsum_k" not in data:
        raise KeyError(f"Prototype dict must contain keys ['features','cumsum_k']. Got keys: {list(data.keys())}")

    feats = data["features"]
    cumsum = data["cumsum_k"]

    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    elif not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)

    feats = feats.float().to(device)
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)

    if isinstance(cumsum, torch.Tensor):
        cumsum = cumsum.detach().cpu().tolist()
    elif isinstance(cumsum, np.ndarray):
        cumsum = cumsum.tolist()
    cumsum = [int(x) for x in cumsum]
    if len(cumsum) != 5:
        raise ValueError(f"cumsum_k must have length 5. Got: {cumsum}")

    if use_all:
        print(f"Loaded prototype BANK: {tuple(feats.shape)} | cumsum={cumsum} | class_order={data.get('class_order', None)}")
        return feats

    protos = []
    for i in range(4):
        s, e = cumsum[i], cumsum[i+1]
        p = feats[s:e].mean(dim=0)
        p = p / (p.norm() + 1e-12)
        protos.append(p)
    protos = torch.stack(protos, dim=0)
    print(f"Loaded per-class mean prototypes: {tuple(protos.shape)} | cumsum={cumsum} | class_order={data.get('class_order', None)}")
    return protos


def generate_csv_from_filenames(data_root, output_path):
    """
    Generate CSV file from image filenames with label encoding
    
    Args:
        data_root: Directory containing images
        output_path: Path to save CSV file
    """
    files = [f for f in os.listdir(data_root) if f.endswith(".png")]
    data = []
    print(f"[CSV] Creating labels for {len(files)} images...")

    for filename in files:
        try:
            label_str = filename.rsplit("[", 1)[1].rsplit("]", 1)[0]
            if len(label_str) != 4:
                print(f"⚠️ Skipping strange file: {filename}")
                continue
            tumor = int(label_str[0])
            stroma = int(label_str[1])
            lymph = int(label_str[2])
            necro = int(label_str[3])

            data.append({
                "image_id": filename.replace(".png", ""),
                "filename": filename,
                "Tumor": tumor,
                "Stroma": stroma,
                "Lymphocytic infiltrate": lymph,
                "Necrosis": necro
            })
        except Exception as e:
            print(f"[CSV] Error parsing {filename}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[CSV] Saved: {output_path} | rows={len(df)}")
    return df


# CONCH normalize stats (fallback to CLIP defaults)
_CLIP_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]


def _extract_normalize_stats_from_preprocess(preprocess):
    """Extract mean and std from preprocess transform"""
    mean, std = None, None
    try:
        if isinstance(preprocess, transforms.Compose):
            for t in preprocess.transforms:
                if isinstance(t, transforms.Normalize):
                    mean = list(map(float, t.mean))
                    std = list(map(float, t.std))
                    return mean, std
    except Exception:
        pass
    return None, None


def get_conch_norm_stats(preprocess):
    """Get CONCH normalization stats, fallback to CLIP defaults"""
    mean, std = _extract_normalize_stats_from_preprocess(preprocess)
    if mean is None or std is None:
        mean, std = _CLIP_DEFAULT_MEAN, _CLIP_DEFAULT_STD
    return mean, std

