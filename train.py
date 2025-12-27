"""
Main training script
"""
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to import CONCH
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
except Exception as e:
    raise ImportError(
        "Cannot import CONCH. Ensure conch.open_clip_custom is available.\n"
        f"Original error: {e}"
    )

# Import local modules
from config_loader import load_config
from utils import set_seed, load_prototypes_cluster_bank, get_conch_norm_stats
from dataloader import WSSS_Dataset, update_conch_norm
import dataloader as dl_module
from model import Model, inject_lora_conch_text, conch_lora_sanity_check, get_lora_parameters
from losses import FocalFrequencyLoss
from metrics import compute_cls_metrics, tune_thresholds_per_class
from eval import eval_classifier, pseudo_miou_on_dir, _print_pseudo_report, visualize_random_samples


@torch.no_grad()
def _norm_cam_for_loss(cam):
    """Normalize CAM for loss computation"""
    cam = F.relu(cam)
    cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return cam


def cam_equivariance_loss(model, imgs, target_size, tau_present=0.2):
    """CAM equivariance loss for data augmentation consistency"""
    logits1, cam1, _, _, _ = model.forward_cam_logits_multiscale(imgs)

    imgs_f = torch.flip(imgs, dims=[3])
    logits2, cam2, _, _, _ = model.forward_cam_logits_multiscale(imgs_f)
    cam2 = torch.flip(cam2, dims=[3])

    cam1 = F.interpolate(cam1, size=(target_size, target_size), mode="bilinear", align_corners=False)
    cam2 = F.interpolate(cam2, size=(target_size, target_size), mode="bilinear", align_corners=False)

    cam1 = _norm_cam_for_loss(cam1)
    cam2 = _norm_cam_for_loss(cam2)

    p = torch.sigmoid(logits1).detach()
    present = (p >= tau_present).float().view(p.size(0), p.size(1), 1, 1)
    return (present * (cam1 - cam2).abs()).mean()


def _collect_trainable_params(module: nn.Module):
    """Collect trainable parameters from module"""
    return [p for p in module.parameters() if p.requires_grad]


def _dedup_params(params):
    """Remove duplicate parameters"""
    seen, uniq = set(), []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq


def _has_any_grad(params):
    """Check if any parameter has gradient"""
    for p in params:
        if p.grad is not None:
            return True
    return False


def train_joint(config_path="config.yaml"):
    """Main training function"""
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["SEED"])
    os.makedirs(config["PLOT_DIR"], exist_ok=True)

    device = config["DEVICE"]
    use_cuda = (device == "cuda")
    print(f"device: {device}")

    # Speed / determinism toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load CONCH
    conch_model, preprocess = create_model_from_pretrained(
        config["CONCH_ARCH"],
        config["CONCH_HF"],
        hf_auth_token=config.get("HF_TOKEN", None),
    )
    conch_model = conch_model.to(device)

    conch_tokenizer = get_tokenizer()

    # Inject LoRA into text tower
    cnt = inject_lora_conch_text(
        conch_model.text,
        r=config["LORA_R"],
        alpha=config["LORA_ALPHA"],
        dropout=config["LORA_DROPOUT"],
        mode=config.get("LORA_TEXT_TARGET_MODE", "attn_mlp"),
    )
    print(f"Injected LoRA into CONCH text tower: replaced {cnt} Linear layers")
    conch_lora_sanity_check(conch_model)

    # Update global CONCH normalization stats
    m, s = get_conch_norm_stats(preprocess)
    update_conch_norm(m, s)
    dl_module.update_conch_norm(m, s)
    print("CONCH norm:", m, s)

    # Prototypes (optional)
    prototypes = None
    if config.get("USE_IMAGE_PROTOS", False):
        prototype_path = config.get("PROTOTYPE_PATH", "")
        
        # Auto-generate prototype if path is empty or file doesn't exist
        if not prototype_path or not os.path.exists(prototype_path):
            if not prototype_path:
                # Default path
                proto_dir = os.path.join(os.path.dirname(config_path), "prototypes")
                os.makedirs(proto_dir, exist_ok=True)
                prototype_path = os.path.join(proto_dir, "bcss_prototypes.pkl")
            
            print(f"Prototype file not found or path empty. Auto-generating: {prototype_path}")
            try:
                # Import generate function
                from generate_prototypes import generate_prototypes
                
                # Get paths from config
                image_dir = config.get("CLS_IMG_DIR", config.get("CSSL_IMG_DIR", ""))
                label_csv = config.get("CLS_LABEL_CSV", "")
                
                if not image_dir or not label_csv:
                    print("Warning: CLS_IMG_DIR or CLS_LABEL_CSV not set. Skipping prototype generation.")
                    prototypes = None
                else:
                    # Generate prototypes with k_list from config or default
                    # Note: Only uses images with exactly 1 class (single-label)
                    k_list = config.get("PROTO_K_LIST", [3, 3, 3, 3])
                    
                    print(f"\n{'='*60}")
                    print("Auto-Generating Image Prototypes")
                    print(f"{'='*60}")
                    print(f"  Image directory: {image_dir}")
                    print(f"  Label CSV: {label_csv}")
                    print(f"  Output path: {prototype_path}")
                    print(f"  K list (clusters per class): {k_list}")
                    print(f"  Filter: Only images with exactly 1 class")
                    print(f"  CONCH arch: {config.get('CONCH_ARCH', 'conch_ViT-B-16')}")
                    print(f"  Target size: {config.get('TARGET_SIZE', 448)}")
                    print(f"{'='*60}\n")
                    
                    generate_prototypes(
                        image_dir=image_dir,
                        label_csv=label_csv,
                        output_path=prototype_path,
                        k_list=k_list,
                        conch_arch=config.get("CONCH_ARCH", "conch_ViT-B-16"),
                        conch_hf=config.get("CONCH_HF", "hf_hub:MahmoodLab/conch"),
                        hf_token=config.get("HF_TOKEN", None),
                        device=device,
                        target_size=config.get("TARGET_SIZE", 448)
                    )
                    print(f"\n✓ Prototype generated successfully: {prototype_path}\n")
            except Exception as e:
                print(f"Error generating prototype: {e}")
                print("Continuing without prototypes...")
                prototypes = None
                prototype_path = None
        
        # Load prototype if path exists
        if prototype_path and os.path.exists(prototype_path):
            print(f"\n{'='*60}")
            print("Loading Image Prototypes")
            print(f"{'='*60}")
            print(f"Prototype path: {prototype_path}")
            print(f"Use all bank: {config.get('PROTO_USE_ALL_BANK', True)}")
            
            prototypes = load_prototypes_cluster_bank(
                prototype_path, device=device, use_all=config.get("PROTO_USE_ALL_BANK", True)
            )
            
            # Print detailed prototype info
            if prototypes is not None:
                import pickle as pkl
                try:
                    with open(prototype_path, 'rb') as f:
                        proto_data = pkl.load(f)
                    
                    if isinstance(proto_data, dict):
                        k_list = proto_data.get('k_list', 'N/A')
                        class_order = proto_data.get('class_order', 'N/A')
                        cumsum_k = proto_data.get('cumsum_k', [])
                        
                        print(f"Prototype details:")
                        print(f"  - Total prototypes: {prototypes.shape[0]}")
                        print(f"  - Feature dimension: {prototypes.shape[1]}")
                        print(f"  - K list (clusters per class): {k_list}")
                        print(f"  - Class order: {class_order}")
                        print(f"  - Cumulative indices: {cumsum_k}")
                        
                        if isinstance(class_order, list) and len(cumsum_k) == 5:
                            print(f"\n  Prototype distribution per class:")
                            for i, class_name in enumerate(class_order):
                                start_idx = cumsum_k[i]
                                end_idx = cumsum_k[i+1]
                                num_protos = end_idx - start_idx
                                print(f"    {class_name:>24s}: {num_protos:>2d} prototypes (indices {start_idx:>2d}-{end_idx-1:>2d})")
                except Exception as e:
                    print(f"  (Could not load metadata: {e})")
            
            print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("Image Prototypes: DISABLED")
        print(f"{'='*60}\n")

    # Build model
    print(f"\n{'='*60}")
    print("Building Model")
    print(f"{'='*60}")
    print(f"Using image prototypes: {prototypes is not None}")
    if prototypes is not None:
        print(f"  Prototype shape: {prototypes.shape}")
        print(f"  Prototype device: {prototypes.device}")
        print(f"  Prototype dtype: {prototypes.dtype}")
    print(f"{'='*60}\n")
    
    model = Model(
        conch_model=conch_model,
        conch_tokenizer=conch_tokenizer,
        config=config,
        prototypes=prototypes,
        num_classes=4,
    ).to(device)

    model.build_text_knowledge_bases(device)
    
    # Build initial fine cache (no_grad) to avoid OOM in step 0
    if config.get("FINE_CACHE_ENABLE", True):
        model.prompt_bank.update_fine_cache(
            model.conch,
            device=torch.device(device),
            dtype=config.get("FINE_CACHE_DTYPE", "fp16"),
            chunk=config.get("FINE_CACHE_CHUNK", 64),
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    # Optimizers
    bce = nn.BCEWithLogitsLoss()

    vision_params = [p for p in model.conch.visual.parameters() if p.requires_grad]
    print(f"Trainable vision params: {sum(p.numel() for p in vision_params):,}")

    lora_params = get_lora_parameters(model.conch.text)
    assert len(lora_params) > 0, "LoRA mandatory but no LoRA params found."
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    head_params = []
    head_params += _collect_trainable_params(model.sim)
    head_params += _collect_trainable_params(model.know_attn1) + _collect_trainable_params(model.know_attn2)
    head_params += _collect_trainable_params(model.knowledge_bank.proj_to_vision)
    head_params += _collect_trainable_params(model.knowledge_bank.proj_to_text)
    head_params += _collect_trainable_params(model.img_to_text)
    head_params += _collect_trainable_params(model.aff_proj)

    if isinstance(getattr(model, "fine_beta", None), torch.nn.Parameter) and model.fine_beta.requires_grad:
        head_params += [model.fine_beta]

    if getattr(model, "use_img_protos", False):
        head_params += _collect_trainable_params(model.proto_proj) + [model.proto_balance]

    head_params = _dedup_params(head_params)
    # remove overlap w/ lora
    lora_set = set(lora_params)
    head_params = [p for p in head_params if p not in lora_set]

    opt_vision = torch.optim.Adam(vision_params, lr=config["LR_VISION"], betas=(0.5, 0.999)) if len(vision_params) else None
    opt_head = torch.optim.Adam(head_params, lr=config["LR_HEAD"], betas=(0.5, 0.999))
    opt_lora = torch.optim.Adam(lora_params, lr=config["LR_LORA"], betas=(0.5, 0.999))

    scaler = GradScaler(enabled=use_cuda)

    # Datasets / loaders
    target_size = int(config["TARGET_SIZE"])
    ds_cls = WSSS_Dataset(
        config["CLS_IMG_DIR"], config["CLS_LABEL_CSV"],
        target_size=target_size, mean=m, std=s
    )

    n_val = int(len(ds_cls) * float(config["VAL_SPLIT"]))
    n_train = len(ds_cls) - n_val
    train_cls, val_cls = random_split(
        ds_cls, [n_train, n_val],
        generator=torch.Generator().manual_seed(config["SEED"])
    )

    cls_train_loader = DataLoader(
        train_cls, batch_size=config["BATCH_CLS"], shuffle=True,
        num_workers=config["NUM_WORKERS"], drop_last=True, pin_memory=use_cuda
    )
    cls_val_loader = DataLoader(
        val_cls, batch_size=config["BATCH_CLS"], shuffle=False,
        num_workers=config["NUM_WORKERS"], drop_last=False, pin_memory=use_cuda
    )

    test_img_dir = config.get("TEST_IMG_DIR", None)
    test_mask_dir = config.get("TEST_MASK_DIR", None)
    has_test = (test_img_dir is not None) and (test_mask_dir is not None) and os.path.isdir(test_img_dir) and os.path.isdir(test_mask_dir)

    best_score = -1e9
    best_epoch = -1
    best_thr = None

    attr_w = float(config.get("ATTR_LOSS_WEIGHT", 0.05))
    accum = int(max(1, config.get("ACCUM_STEPS", 1)))

    eval_config = config.get("EVAL", {})

    for epoch in range(int(config["EPOCHS"])):
        print(f"\n=== Epoch {epoch+1}/{config['EPOCHS']} ===")
        
        # Refresh fine cache per epoch (so it tracks LoRA updates, still no_grad)
        if config.get("FINE_CACHE_ENABLE", True) and config.get("FINE_CACHE_REFRESH", "epoch") == "epoch":
            model.prompt_bank.update_fine_cache(
                model.conch,
                device=torch.device(device),
                dtype=config.get("FINE_CACHE_DTYPE", "fp16"),
                chunk=config.get("FINE_CACHE_CHUNK", 64),
            )
            if device == "cuda":
                torch.cuda.empty_cache()

        model.train()

        running_loss, n_steps = 0.0, 0

        if opt_vision is not None:
            opt_vision.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)
        opt_lora.zero_grad(set_to_none=True)

        loop = tqdm(cls_train_loader, desc="TRAIN", leave=False)
        for step, (imgs, labels) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_cuda):
                fused_logits, fused_cam, (l2, l3, l4), _, extras = model.forward_cam_logits_multiscale(imgs)

                w2, w3, w4 = config["MS_WEIGHTS"]["L2"], config["MS_WEIGHTS"]["L3"], config["MS_WEIGHTS"]["L4"]
                loss_ms = (w2 * bce(l2, labels) + w3 * bce(l3, labels) + w4 * bce(l4, labels)) / (w2 + w3 + w4)

                lam = 0.15 * min(1.0, (epoch + 1) / 3.0)
                loss_eq = cam_equivariance_loss(model, imgs, target_size, tau_present=0.2)

                loss_attr = 0.0
                if (attr_w > 0) and extras.get("has_fine", False):
                    attr_logits = extras.get("attr_logits", None)
                    if attr_logits is not None:
                        loss_attr = F.binary_cross_entropy_with_logits(attr_logits, labels)

                total_loss = (loss_ms + lam * loss_eq + attr_w * loss_attr) / float(accum)

            scaler.scale(total_loss).backward()

            if ((step + 1) % accum == 0):
                # unscale
                if opt_vision is not None:
                    scaler.unscale_(opt_vision)
                scaler.unscale_(opt_head)
                scaler.unscale_(opt_lora)

                # clip
                if opt_vision is not None and len(vision_params):
                    torch.nn.utils.clip_grad_norm_(vision_params, max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=5.0)

                # Safe step: only if has grads (prevents GradScaler assertion)
                if opt_vision is not None and _has_any_grad(vision_params):
                    scaler.step(opt_vision)
                if _has_any_grad(head_params):
                    scaler.step(opt_head)
                if _has_any_grad(lora_params):
                    scaler.step(opt_lora)

                scaler.update()

                if opt_vision is not None:
                    opt_vision.zero_grad(set_to_none=True)
                opt_head.zero_grad(set_to_none=True)
                opt_lora.zero_grad(set_to_none=True)

            running_loss += float(total_loss.item()) * float(accum)
            n_steps += 1
            loop.set_postfix({"Loss": running_loss / max(n_steps, 1)})

        # CLS validation + tune thresholds
        val_res, val_df, y_true, y_prob = eval_classifier(
            model, cls_val_loader, device, threshold=config["THRESH_DEFAULT"],
            class_names=config["CLASS_NAMES"]
        )

        tuned_thr = None
        if config["TUNE_THRESHOLDS"]:
            tuned_thr = tune_thresholds_per_class(y_true, y_prob, grid_points=config["THRESH_GRID_POINTS"])
            valT_res, _ = compute_cls_metrics(y_true, y_prob, tuned_thr, class_names=config["CLASS_NAMES"])
            eval_config["THR_VEC"] = tuned_thr.tolist()
            print(f"[CLS] Ep {epoch+1}: Loss={val_res['loss']:.4f} | mAP(tune)={valT_res['map_macro']:.4f}")
        else:
            print(f"[CLS] Ep {epoch+1}: Loss={val_res['loss']:.4f} | mAP={val_res['map_macro']:.4f}")

        # PSEUDO VAL: auto-tau + update global tau
        best_val = pseudo_miou_on_dir(
            model, device, config["VAL_IMG_DIR"], config["VAL_MASK_DIR"],
            tag="VAL",
            auto_tau=True,
            update_global_tau=True,
            tau_candidates=eval_config.get("TAU_CANDIDATES", None),
            eval_config=eval_config,
            class_names=config["CLASS_NAMES"],
            target_size=target_size,
        )
        _print_pseudo_report("VAL", best_val, config["CLASS_NAMES"])

        # PSEUDO TEST: fixed tau from VAL
        best_test = None
        if has_test:
            best_test = pseudo_miou_on_dir(
                model, device, test_img_dir, test_mask_dir,
                tag="TEST",
                auto_tau=False,
                update_global_tau=False,
                fixed_tau=float(eval_config["PIXEL_TAU"]),
                eval_config=eval_config,
                class_names=config["CLASS_NAMES"],
                target_size=target_size,
            )
            _print_pseudo_report("TEST", best_test, config["CLASS_NAMES"])
        else:
            print("[TEST] PSEUDO: (skip — TEST_IMG_DIR/TEST_MASK_DIR not set or not found)")

        # Visualize random samples
        n_vis = int(eval_config.get("VIS_SAMPLES", 2))
        if n_vis > 0 and best_val is not None:
            visualize_random_samples(
                model, device, config["VAL_IMG_DIR"], config["VAL_MASK_DIR"],
                tag="VAL", tau_use=best_val["tau"], n_samples=n_vis,
                seed=int(config.get("SEED", 123)) + epoch * 17,
                eval_config=eval_config,
                class_names=config["CLASS_NAMES"],
                target_size=target_size,
            )
            if has_test and (best_test is not None):
                visualize_random_samples(
                    model, device, test_img_dir, test_mask_dir,
                    tag="TEST", tau_use=best_test["tau"], n_samples=n_vis,
                    seed=int(config.get("SEED", 123)) + epoch * 29,
                    eval_config=eval_config,
                    class_names=config["CLASS_NAMES"],
                    target_size=target_size,
                )

        # Choose save criterion
        pseudo_miou_fg = best_val["miou_fg"] if best_val is not None else -1.0
        cur = float(pseudo_miou_fg) if config["SAVE_BEST_BY"] == "pseudo_miou_fg" else float(val_res["map_macro"])

        if cur > best_score:
            best_score = cur
            best_epoch = epoch + 1
            best_thr = tuned_thr.copy() if tuned_thr is not None else None

            ckpt = {
                "model_state": model.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_thresholds": (best_thr.tolist() if best_thr is not None else None),
                "config": config,
                "best_val_pseudo": best_val,
                "best_test_pseudo": best_test,
                "pixel_tau_val": float(eval_config["PIXEL_TAU"]),
                "conch_norm": {"mean": m, "std": s},
            }
            torch.save(ckpt, config["SAVE_PATH"])
            print(f"Saved BEST (Ep {best_epoch}, Score={best_score:.4f}) -> {config['SAVE_PATH']}")

    return config["SAVE_PATH"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BCSS-WSSS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    ckpt_path = train_joint(args.config)
    print("Saved:", ckpt_path)

