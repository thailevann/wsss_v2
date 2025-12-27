"""
Evaluation functions: pseudo mIoU, visualization, TTA
"""
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from dataloader import BCSSValDataset
from metrics import compute_cls_metrics


@torch.no_grad()
def eval_classifier(model, loader, device, threshold=0.5, class_names=None, show_pbar=True):
    """Evaluate classifier on validation set"""
    model.eval()
    all_logits, all_labels = [], []
    loss_sum, n = 0.0, 0
    criterion = nn.BCEWithLogitsLoss()

    it = tqdm(loader, desc="VAL-CLS", leave=False) if show_pbar else loader
    for imgs, labels in it:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        fused_logits, _, _, _, _ = model.forward_cam_logits_multiscale(imgs)
        loss = criterion(fused_logits, labels)

        bs = imgs.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        all_logits.append(fused_logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if show_pbar:
            it.set_postfix({"loss": loss_sum / max(n, 1)})

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)

    summary, df = compute_cls_metrics(y_true, probs, threshold, class_names=class_names)
    summary["loss"] = loss_sum / max(n, 1)
    return summary, df, y_true, probs


def compute_tissue_mask(img_rgb, target_hw=None, eval_config=None):
    """
    Compute tissue mask from RGB image
    img_rgb: numpy HxWx3 uint8 OR torch uint8 HxWx3
    target_hw: (H,W) => resize mask to match pred/gt
    returns: uint8 HxW (0/1)
    """
    if eval_config is None:
        eval_config = {
            "TISSUE_S_MIN": 15,
            "TISSUE_V_MAX": 240,
            "TISSUE_MORPH": 3,
        }
    
    if torch.is_tensor(img_rgb):
        img_rgb = img_rgb.detach().cpu().numpy()
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    img_rgb = np.ascontiguousarray(img_rgb)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    tissue = ((s >= eval_config["TISSUE_S_MIN"]) | (v <= eval_config["TISSUE_V_MAX"])).astype(np.uint8)

    k = int(eval_config["TISSUE_MORPH"])
    if k and k > 1:
        kernel = np.ones((k, k), np.uint8)
        tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=1)
        tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel, iterations=1)

    if target_hw is not None:
        th, tw = int(target_hw[0]), int(target_hw[1])
        if tissue.shape[0] != th or tissue.shape[1] != tw:
            tissue = cv2.resize(tissue, (tw, th), interpolation=cv2.INTER_NEAREST)

    return tissue


def robust_norm(cam_relu, p=99.0, eps=1e-8):
    """Robust normalization of CAM"""
    if float(p) >= 99.999:
        mx = cam_relu.flatten(2).max(dim=-1)[0].view(cam_relu.size(0), cam_relu.size(1), 1, 1).clamp_min(eps)
        return (cam_relu / mx).clamp(0, 1)
    B, C, H, W = cam_relu.shape
    flat = cam_relu.view(B, C, -1)
    q = torch.quantile(flat, q=float(p)/100.0, dim=-1, keepdim=True).clamp_min(eps)
    out = (flat / q).clamp(0, 1).view(B, C, H, W)
    return out


@torch.no_grad()
def forward_tta_cam(model, imgs, eval_config=None):
    """Test-time augmentation forward pass"""
    if eval_config is None:
        eval_config = {
            "USE_TTA": True,
            "TTA_SCALES": [0.75, 1.00, 1.25],
            "TTA_HFLIP": True,
        }
    
    if not eval_config.get("USE_TTA", True):
        fused_logits, fused_cam, _, _, extras = model.forward_cam_logits_multiscale(imgs)
        return fused_logits, fused_cam, extras

    scales = eval_config.get("TTA_SCALES", [1.0])
    hflip = bool(eval_config.get("TTA_HFLIP", True))

    model.eval()
    B, C, H, W = imgs.shape

    logits_sum, cam_sum = None, None
    attr_sum, feat_sum = None, None
    n = 0

    base_cam_hw, base_feat_hw = None, None

    def add_view(x, unflip=False):
        nonlocal logits_sum, cam_sum, attr_sum, feat_sum, n, base_cam_hw, base_feat_hw

        lg, cam, _, _, extras = model.forward_cam_logits_multiscale(x)

        if unflip:
            cam = torch.flip(cam, dims=[3])
        if base_cam_hw is None:
            base_cam_hw = cam.shape[-2:]
        cam = F.interpolate(cam, size=base_cam_hw, mode="bilinear", align_corners=False)

        ac = extras.get("attr_conf", None)
        feat = extras.get("feat_aff", None)
        if feat is not None:
            if unflip:
                feat = torch.flip(feat, dims=[3])
            if base_feat_hw is None:
                base_feat_hw = feat.shape[-2:]
            feat = F.interpolate(feat, size=base_feat_hw, mode="bilinear", align_corners=False)

        if logits_sum is None:
            logits_sum = lg
            cam_sum = cam
            attr_sum = ac if ac is not None else None
            feat_sum = feat if feat is not None else None
        else:
            logits_sum = logits_sum + lg
            cam_sum = cam_sum + cam
            if attr_sum is not None and ac is not None:
                attr_sum = attr_sum + ac
            if feat_sum is not None and feat is not None:
                feat_sum = feat_sum + feat
        n += 1

    add_view(imgs, unflip=False)

    for s in scales:
        if abs(float(s) - 1.0) < 1e-6:
            continue
        hs, ws = int(round(H * float(s))), int(round(W * float(s)))
        x = F.interpolate(imgs, size=(hs, ws), mode="bilinear", align_corners=False)
        add_view(x, unflip=False)

    if hflip:
        x = torch.flip(imgs, dims=[3])
        add_view(x, unflip=True)

    extras_out = {
        "attr_conf": (attr_sum / max(n, 1) if attr_sum is not None else None),
        "feat_aff": (feat_sum / max(n, 1) if feat_sum is not None else None),
    }
    return logits_sum / max(n, 1), cam_sum / max(n, 1), extras_out


@torch.no_grad()
def affinity_propagate_cam(cam_khw, feat_chw, k=7, iters=2, gamma=2.0, eps=1e-6):
    """Affinity propagation for CAM refinement"""
    assert cam_khw.dim() == 4 and feat_chw.dim() == 4
    B, Kc, H, W = cam_khw.shape
    _, C, Hf, Wf = feat_chw.shape
    assert H == Hf and W == Wf, "cam and feat must have same H,W for affinity."

    k = int(k)
    pad = k // 2
    iters = int(iters)
    gamma = float(gamma)

    feat = feat_chw.float()
    feat = feat / (feat.norm(dim=1, keepdim=True) + eps)

    KK = k * k
    feat_unf = F.unfold(feat, kernel_size=k, padding=pad)  # [B, C*KK, N]
    N = feat_unf.shape[-1]
    feat_unf = feat_unf.view(B, C, KK, N)
    feat_ctr = feat.view(B, C, N).unsqueeze(2)

    sim = (feat_unf * feat_ctr).sum(dim=1)     # [B,KK,N]
    sim = torch.clamp(sim, min=0.0) ** gamma
    sim = sim / (sim.sum(dim=1, keepdim=True) + eps)

    cam = cam_khw.float()
    for _ in range(iters):
        cam_unf = F.unfold(cam, kernel_size=k, padding=pad).view(B, Kc, KK, N)
        cam_new = (cam_unf * sim.unsqueeze(1)).sum(dim=2)  # [B,K,N]
        cam = cam_new.view(B, Kc, H, W)
        cam = torch.clamp(cam, 0.0, 1e9)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + eps)
    return cam


@torch.no_grad()
def _present_vector(logits_1x4, cam_norm_1x4hw, attr_conf_1x4=None, eval_config=None):
    """Compute present vector for classes"""
    if eval_config is None:
        eval_config = {
            "THR_VEC": [0.40, 0.40, 0.35, 0.40],
            "PEAK_VEC": [0.85, 0.90, 0.80, 0.85],
            "HARD_MIN": [0.05, 0.05, 0.03, 0.03],
            "USE_ATTR_PRESENT": True,
            "ATTR_THR_VEC": [0.55, 0.55, 0.55, 0.55],
        }
    
    prob = torch.sigmoid(logits_1x4)[0].detach().cpu().numpy()
    peak = cam_norm_1x4hw.flatten(2).max(dim=-1)[0][0].detach().cpu().numpy()

    thr = np.array(eval_config["THR_VEC"], dtype=np.float32)
    pk = np.array(eval_config["PEAK_VEC"], dtype=np.float32)
    hard = np.array(eval_config["HARD_MIN"], dtype=np.float32)

    present = np.zeros(4, dtype=np.float32)
    for c in range(4):
        if prob[c] < hard[c]:
            present[c] = 0.0
        elif (prob[c] >= thr[c]) or (peak[c] >= pk[c]):
            present[c] = 1.0

    if eval_config.get("USE_ATTR_PRESENT", True) and (attr_conf_1x4 is not None):
        attr_thr = np.array(eval_config.get("ATTR_THR_VEC", [0.55]*4), dtype=np.float32)
        ac = attr_conf_1x4[0].detach().cpu().numpy()
        for c in range(4):
            if ac[c] >= attr_thr[c]:
                present[c] = 1.0
    return present


@torch.no_grad()
def cams_to_mask_affinity(img_rgb, logits_1x4, cam_1x4hw, feat_aff_1xChw, tau_lo, 
                          attr_conf_1x4=None, out_hw=None, eval_config=None):
    """Convert CAMs to mask using affinity propagation"""
    if eval_config is None:
        eval_config = {
            "SEED_DELTA": 0.22,
            "ROBUST_P": 99.0,
            "CLASS_SCALE": [1.00, 0.95, 1.15, 1.10],
            "AFF_K": 7,
            "AFF_ITERS": 2,
            "AFF_GAMMA": 2.0,
            "AFF_EPS": 1e-6,
            "USE_TISSUE_MASK": True,
        }
    
    device = cam_1x4hw.device
    tau_lo = float(tau_lo)
    tau_hi = float(min(0.95, tau_lo + float(eval_config.get("SEED_DELTA", 0.22))))

    if out_hw is None:
        out_h, out_w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
    else:
        out_h, out_w = int(out_hw[0]), int(out_hw[1])

    Hf, Wf = feat_aff_1xChw.shape[-2:]
    cam = F.interpolate(cam_1x4hw, size=(Hf, Wf), mode="bilinear", align_corners=False)
    cam = F.relu(cam)
    cam = robust_norm(cam, p=float(eval_config["ROBUST_P"]), eps=1e-8)

    present = _present_vector(logits_1x4, cam, attr_conf_1x4=attr_conf_1x4, eval_config=eval_config)
    present_t = torch.tensor(present, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    cam = cam * present_t

    scale = torch.tensor(eval_config["CLASS_SCALE"], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    cam = cam * scale

    cam_ref = affinity_propagate_cam(
        cam, feat_aff_1xChw,
        k=int(eval_config["AFF_K"]),
        iters=int(eval_config["AFF_ITERS"]),
        gamma=float(eval_config["AFF_GAMMA"]),
        eps=float(eval_config["AFF_EPS"]),
    )

    score_max, score_arg = cam_ref.max(dim=1)  # [1,Hf,Wf]
    pred = (score_arg + 1).byte()
    pred[score_max < tau_lo] = 0

    seed_mask = (score_max >= tau_hi)
    pred_seed = (score_arg + 1).byte()
    pred[seed_mask] = pred_seed[seed_mask]

    pred_up = F.interpolate(pred.float().unsqueeze(1), size=(out_h, out_w), mode="nearest").squeeze(1).byte()
    pred_np = pred_up[0].detach().cpu().numpy()

    if eval_config.get("USE_TISSUE_MASK", True):
        tissue = compute_tissue_mask(img_rgb, target_hw=pred_np.shape[:2], eval_config=eval_config)
        pred_np[tissue == 0] = 0
    return pred_np


@torch.no_grad()
def cams_to_mask_simple(img_rgb, logits_1x4, cam_1x4hw, pixel_tau=None, 
                        attr_conf_1x4=None, out_hw=None, eval_config=None):
    """Convert CAMs to mask (simple method)"""
    if eval_config is None:
        eval_config = {
            "PIXEL_TAU": 0.18,
            "ROBUST_P": 99.0,
            "CLASS_SCALE": [1.00, 0.95, 1.15, 1.10],
            "USE_TISSUE_MASK": True,
        }
    
    tau = float(eval_config["PIXEL_TAU"] if pixel_tau is None else pixel_tau)
    device = cam_1x4hw.device

    if out_hw is None:
        out_h, out_w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
    else:
        out_h, out_w = int(out_hw[0]), int(out_hw[1])

    cam_up = F.interpolate(cam_1x4hw, size=(out_h, out_w), mode="bilinear", align_corners=False)
    cam_relu = F.relu(cam_up)
    cam_norm = robust_norm(cam_relu, p=float(eval_config["ROBUST_P"]), eps=1e-8)

    present = _present_vector(logits_1x4, cam_norm, attr_conf_1x4=attr_conf_1x4, eval_config=eval_config)
    present_t = torch.tensor(present, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    cam_norm = cam_norm * present_t

    scale = torch.tensor(eval_config["CLASS_SCALE"], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    score = cam_norm * scale

    score_max, score_arg = score.max(dim=1)
    pred = (score_arg + 1).byte()[0].detach().cpu().numpy()
    pred[score_max[0].detach().cpu().numpy() < tau] = 0

    if eval_config.get("USE_TISSUE_MASK", True):
        tissue = compute_tissue_mask(img_rgb, target_hw=pred.shape[:2], eval_config=eval_config)
        pred[tissue == 0] = 0
    return pred


class ConfusionMatrixAllClass(object):
    """Confusion matrix for multi-class segmentation"""
    
    def __init__(self, num_classes):
        self.num_classes = int(num_classes)  # 5 (BG+4)
        self.mat1 = None
        self.mat2 = None  # 2x2

    def update(self, a, b):
        """Update confusion matrix"""
        n = self.num_classes
        if self.mat1 is None:
            self.mat1 = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        if self.mat2 is None:
            self.mat2 = torch.zeros((2, 2), dtype=torch.int64, device=a.device)

        with torch.no_grad():
            k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
            a1 = a[k].to(torch.int64).reshape(-1)
            b1 = b[k].to(torch.int64).reshape(-1)
            inds = (n * a1 + b1)
            self.mat1 += torch.bincount(inds, minlength=n*n).reshape(n, n)

            a_bin = (a != 0).to(torch.int64)
            b_bin = (b != 0).to(torch.int64)
            k2 = (a_bin >= 0) & (a_bin < 2) & (b_bin >= 0) & (b_bin < 2)
            a2 = a_bin[k2].reshape(-1)
            b2 = b_bin[k2].reshape(-1)
            inds2 = (2 * a2 + b2).to(torch.int64).reshape(-1)
            self.mat2 += torch.bincount(inds2, minlength=4).reshape(2, 2)

    def compute(self):
        """Compute metrics from confusion matrix"""
        h = self.mat1.float()
        acc_global = torch.diag(h).sum() / (h.sum() + 1e-12)
        acc = torch.diag(h) / (h.sum(1) + 1e-12)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-12)
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0) + 1e-12)

        h_bg_fg = self.mat2.float()
        dice_bg_fg = 2 * torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0) + 1e-12)
        return acc_global, acc, iu, dice, dice_bg_fg


def _fmt_iou_per_class(iu_5, class_names):
    """Format IoU per class"""
    return {
        "BG": float(iu_5[0]),
        class_names[0]: float(iu_5[1]),
        class_names[1]: float(iu_5[2]),
        class_names[2]: float(iu_5[3]),
        class_names[3]: float(iu_5[4]),
        "mIoU_FG": float(np.mean(iu_5[1:])),
    }


def _print_pseudo_report(tag, best, class_names):
    """Print pseudo mIoU report"""
    if best is None:
        print(f"[{tag}] PSEUDO: None")
        return
    iou = best["iu"]
    fmt = _fmt_iou_per_class(iou, class_names)
    print(f"[{tag}] PSEUDO | mode={best['mode']} | tau={best['tau']:.3f} | mIoU_FG={best['miou_fg']:.2f}")
    print("    ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in fmt.items()})


@torch.no_grad()
def pseudo_miou_on_dir(model, device, img_dir, mask_dir, tag="VAL", 
                       auto_tau=False, update_global_tau=False, 
                       tau_candidates=None, fixed_tau=None, 
                       eval_config=None, class_names=None, target_size=448):
    """Evaluate pseudo mIoU on directory"""
    if class_names is None:
        class_names = ["Tumor", "Stroma", "Lymph", "Necrosis"]
    
    if eval_config is None:
        eval_config = {
            "BATCH_SIZE": 16,
            "NUM_WORKERS": 2,
            "PSEUDO_MODE": "affinity",
            "PIXEL_TAU": 0.18,
            "TAU_CANDIDATES": [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26],
        }
    else:
        # Normalize keys: convert all keys to uppercase for compatibility
        # Handle both lowercase (from config.yaml) and uppercase (default) keys
        normalized = {}
        for k, v in eval_config.items():
            if isinstance(k, str):
                normalized[k.upper()] = v
            else:
                normalized[k] = v
        eval_config = normalized
    
    # Ensure required keys exist with defaults
    eval_config.setdefault("BATCH_SIZE", 16)
    eval_config.setdefault("NUM_WORKERS", 2)
    eval_config.setdefault("PSEUDO_MODE", "affinity")
    eval_config.setdefault("PIXEL_TAU", 0.18)
    eval_config.setdefault("TAU_CANDIDATES", [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26])
    
    ds = BCSSValDataset(img_dir, mask_dir, target_size=target_size)
    if len(ds) == 0:
        print(f"[{tag}] dataset empty.")
        return None

    loader = DataLoader(
        ds, batch_size=eval_config["BATCH_SIZE"], shuffle=False,
        num_workers=eval_config["NUM_WORKERS"], pin_memory=True
    )

    if tau_candidates is None:
        tau_candidates = eval_config.get("TAU_CANDIDATES", [float(eval_config["PIXEL_TAU"])])

    if auto_tau:
        taus = [float(t) for t in tau_candidates]
    else:
        if fixed_tau is None:
            fixed_tau = float(eval_config["PIXEL_TAU"])
        taus = [float(fixed_tau)]

    cms = {t: ConfusionMatrixAllClass(num_classes=5) for t in taus}
    mode = str(eval_config.get("PSEUDO_MODE", "affinity")).lower()

    model.eval()

    outer = tqdm(loader, desc=f"{tag}->PSEUDO mIoU ({'AUTO-TAU' if auto_tau else 'FIX-TAU'})", leave=False)
    for batch in outer:
        imgs = batch["img_tensor"].to(device, non_blocking=True)

        img_rgbs = batch["img_rgb"]
        img_rgbs = img_rgbs.cpu().numpy() if torch.is_tensor(img_rgbs) else np.array(img_rgbs)
        if img_rgbs.dtype != np.uint8:
            img_rgbs = img_rgbs.astype(np.uint8)

        gts = batch["gt"]
        gts = gts.to(device, dtype=torch.int64, non_blocking=True) if torch.is_tensor(gts) \
              else torch.tensor(gts, device=device, dtype=torch.int64)

        fused_logits, fused_cam, extras = forward_tta_cam(model, imgs, eval_config=eval_config)
        attr_conf = extras.get("attr_conf", None)
        feat_aff = extras.get("feat_aff", None)

        if mode == "affinity":
            assert feat_aff is not None, "feat_aff is None. Ensure model returns extras['feat_aff']."

        B = imgs.size(0)
        out_hw = tuple(gts.shape[-2:])  # match GT (e.g., 448x448)

        inner = tqdm(total=len(taus) * B, desc=f"{tag} sweep(batch)", leave=False)

        for t in taus:
            preds = []
            for i in range(B):
                ac_i = attr_conf[i:i+1] if attr_conf is not None else None
                if mode == "affinity":
                    fa_i = feat_aff[i:i+1].to(device)
                    pred = cams_to_mask_affinity(
                        img_rgbs[i], fused_logits[i:i+1], fused_cam[i:i+1],
                        fa_i, tau_lo=float(t), attr_conf_1x4=ac_i, out_hw=out_hw, eval_config=eval_config
                    )
                else:
                    pred = cams_to_mask_simple(
                        img_rgbs[i], fused_logits[i:i+1], fused_cam[i:i+1],
                        pixel_tau=float(t), attr_conf_1x4=ac_i, out_hw=out_hw, eval_config=eval_config
                    )
                preds.append(pred)
                inner.update(1)

            pred_tensor = torch.tensor(np.array(preds), device=device, dtype=torch.int64)
            cms[t].update(gts, pred_tensor)

        inner.close()

    best = None
    for t in taus:
        acc_global, acc, iu, dice, dice_bg_fg = cms[t].compute()
        miou_fg = iu[1:].mean().item() * 100.0
        if (best is None) or (miou_fg > best["miou_fg"]):
            best = {
                "tau": float(t),
                "miou_fg": float(miou_fg),
                "acc_global": float(acc_global.item() * 100.0),
                "iu": (iu.detach().cpu().numpy() * 100.0),
                "dice": (dice.detach().cpu().numpy() * 100.0),
                "dice_bg_fg": (dice_bg_fg.detach().cpu().numpy() * 100.0),
                "mode": mode,
                "auto_tau": bool(auto_tau),
            }

    if update_global_tau and best is not None:
        eval_config["PIXEL_TAU"] = float(best["tau"])

    return best


# Visualization helpers
_PALETTE = {
    0: (255, 255, 255),  # BG
    1: (255, 0, 0),      # TUM
    2: (0, 255, 0),      # STR
    3: (0, 0, 255),      # LYM
    4: (153, 0, 255),    # NEC
}


def colorize_mask(mask_hw):
    """Colorize mask for visualization"""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, col in _PALETTE.items():
        out[mask_hw == k] = col
    return out


def overlay_mask(img_rgb, mask_hw, alpha=0.45):
    """Overlay mask on image"""
    col = colorize_mask(mask_hw)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


def cam_to_heat_overlay(img_rgb, cam_hw, thr=None, alpha=0.50):
    """Convert CAM to heatmap overlay"""
    cam_hw = np.ascontiguousarray(cam_hw)
    heat = (cam_hw * 255.0).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    if thr is not None:
        m = (cam_hw >= float(thr)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out2 = out.copy()
        cv2.drawContours(out2, cnts, -1, (255, 255, 255), 1)
        out = out2
    return out


@torch.no_grad()
def visualize_random_samples(model, device, img_dir, mask_dir, tag="VAL", 
                             tau_use=None, n_samples=2, seed=123, 
                             eval_config=None, class_names=None, target_size=448):
    """Visualize random samples"""
    if class_names is None:
        class_names = ["Tumor", "Stroma", "Lymph", "Necrosis"]
    
    if eval_config is None:
        eval_config = {
            "PSEUDO_MODE": "affinity",
            "PIXEL_TAU": 0.18,
            "SEED_DELTA": 0.22,
            "ROBUST_P": 99.0,
            "USE_TISSUE_MASK": True,
            "VIS_FIGSIZE": (18, 9),
        }
    else:
        # Normalize keys: convert all keys to uppercase for compatibility
        normalized = {}
        for k, v in eval_config.items():
            if isinstance(k, str):
                normalized[k.upper()] = v
            else:
                normalized[k] = v
        eval_config = normalized
    
    ds = BCSSValDataset(img_dir, mask_dir, target_size=target_size)
    if len(ds) == 0:
        print(f"[{tag}] empty for visualization.")
        return

    rng = random.Random(int(seed))
    idxs = [rng.randrange(0, len(ds)) for _ in range(int(n_samples))]

    mode = str(eval_config.get("PSEUDO_MODE", "affinity")).lower()
    tau_lo = float(eval_config["PIXEL_TAU"] if tau_use is None else tau_use)
    tau_hi = float(min(0.95, tau_lo + float(eval_config.get("SEED_DELTA", 0.22))))

    for j, idx in enumerate(idxs):
        item = ds[idx]
        img_t = item["img_tensor"].unsqueeze(0).to(device)

        img_rgb = item["img_rgb"]
        img_rgb = img_rgb.cpu().numpy() if torch.is_tensor(img_rgb) else np.array(img_rgb)
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)

        gt = item["gt"]
        gt = gt.cpu().numpy() if torch.is_tensor(gt) else np.array(gt)

        out_hw = img_rgb.shape[:2]

        fused_logits, fused_cam, extras = forward_tta_cam(model, img_t, eval_config=eval_config)
        attr_conf = extras.get("attr_conf", None)
        feat_aff = extras.get("feat_aff", None)

        if mode == "affinity" and feat_aff is not None:
            pred = cams_to_mask_affinity(img_rgb, fused_logits, fused_cam, feat_aff, 
                                        tau_lo=tau_lo, attr_conf_1x4=attr_conf, 
                                        out_hw=out_hw, eval_config=eval_config)
        else:
            pred = cams_to_mask_simple(img_rgb, fused_logits, fused_cam, 
                                      pixel_tau=tau_lo, attr_conf_1x4=attr_conf, 
                                      out_hw=out_hw, eval_config=eval_config)

        # CAM resized to img size for overlay
        cam_up = F.interpolate(fused_cam, size=out_hw, mode="bilinear", align_corners=False)
        cam_up = F.relu(cam_up)
        cam_norm = robust_norm(cam_up, p=float(eval_config["ROBUST_P"]), eps=1e-8)[0].detach().cpu().numpy()  # [4,H,W]

        gt_overlay = overlay_mask(img_rgb, gt, alpha=0.45)
        pred_overlay = overlay_mask(img_rgb, pred, alpha=0.45)

        cam_overlays = []
        for c in range(4):
            cam_overlays.append(cam_to_heat_overlay(img_rgb, cam_norm[c], thr=tau_hi, alpha=0.50))

        fig = plt.figure(figsize=eval_config.get("VIS_FIGSIZE", (18, 9)))
        fig.suptitle(f"[{tag}] sample {j+1}/{len(idxs)} | mode={mode} | tau_lo={tau_lo:.2f} tau_hi={tau_hi:.2f}", fontsize=12)

        ax = plt.subplot(2, 4, 1); ax.imshow(img_rgb); ax.set_title("RGB"); ax.axis("off")
        ax = plt.subplot(2, 4, 2); ax.imshow(gt_overlay); ax.set_title("GT overlay"); ax.axis("off")
        ax = plt.subplot(2, 4, 3); ax.imshow(pred_overlay); ax.set_title("Pseudo overlay"); ax.axis("off")

        if eval_config.get("USE_TISSUE_MASK", True):
            tissue = compute_tissue_mask(img_rgb, target_hw=out_hw, eval_config=eval_config)
            ax = plt.subplot(2, 4, 4); ax.imshow(tissue, cmap="gray"); ax.set_title("Tissue mask"); ax.axis("off")
        else:
            ax = plt.subplot(2, 4, 4); ax.imshow(np.zeros(out_hw, dtype=np.uint8)); ax.set_title("Tissue mask (off)"); ax.axis("off")

        for c in range(4):
            ax = plt.subplot(2, 4, 5+c)
            ax.imshow(cam_overlays[c])
            ax.set_title(f"CAM {class_names[c]} (mark@tau_hi)")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

