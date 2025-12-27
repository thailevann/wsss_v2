"""
Quick test script for BCSS-WSSS model
Tests forward pass, eval functions without full training
"""
import os
import sys
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from dataloader import WSSS_Dataset, BCSSValDataset
from model import Model, inject_lora_conch_text, conch_lora_sanity_check
from eval import eval_classifier, pseudo_miou_on_dir, forward_tta_cam
from metrics import compute_cls_metrics


def test_model_forward(model, device, batch_size=2, target_size=448):
    """Test forward pass of model"""
    print("\n" + "="*60)
    print("TEST: Forward Pass")
    print("="*60)
    
    # Create dummy input
    dummy_imgs = torch.randn(batch_size, 3, target_size, target_size).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            fused_logits, fused_cam, (l2, l3, l4), (cam2, cam3, cam4), extras = \
                model.forward_cam_logits_multiscale(dummy_imgs)
            
            print(f"✓ Forward pass successful")
            print(f"  - Fused logits shape: {fused_logits.shape}")
            print(f"  - Fused CAM shape: {fused_cam.shape}")
            print(f"  - Multi-scale logits: l2={l2.shape}, l3={l3.shape}, l4={l4.shape}")
            print(f"  - Multi-scale CAMs: cam2={cam2.shape}, cam3={cam3.shape}, cam4={cam4.shape}")
            print(f"  - Extras keys: {list(extras.keys())}")
            
            if "attr_logits" in extras:
                print(f"  - Attr logits shape: {extras['attr_logits'].shape}")
            if "gate" in extras:
                print(f"  - Gate shape: {extras['gate'].shape}")
            if "feat_aff" in extras:
                print(f"  - Affinity features shape: {extras['feat_aff'].shape}")
                
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_tta(model, device, batch_size=1, target_size=448):
    """Test TTA forward pass"""
    print("\n" + "="*60)
    print("TEST: TTA Forward Pass")
    print("="*60)
    
    dummy_imgs = torch.randn(batch_size, 3, target_size, target_size).to(device)
    
    eval_config = {
        "USE_TTA": True,
        "TTA_SCALES": [0.75, 1.0, 1.25],
        "TTA_HFLIP": True,
    }
    
    model.eval()
    with torch.no_grad():
        try:
            fused_logits, fused_cam, extras = forward_tta_cam(model, dummy_imgs, eval_config=eval_config)
            
            print(f"✓ TTA forward pass successful")
            print(f"  - Fused logits shape: {fused_logits.shape}")
            print(f"  - Fused CAM shape: {fused_cam.shape}")
            return True
        except Exception as e:
            print(f"✗ TTA forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_classification_eval(model, device, config, preprocess, max_samples=10):
    """Test classification evaluation on small subset"""
    print("\n" + "="*60)
    print("TEST: Classification Evaluation")
    print("="*60)
    
    target_size = int(config["TARGET_SIZE"])
    m, s = get_conch_norm_stats(preprocess)
    
    # Create small dataset
    try:
        ds_cls = WSSS_Dataset(
            config["CLS_IMG_DIR"], config["CLS_LABEL_CSV"],
            target_size=target_size, mean=m, std=s
        )
        
        if len(ds_cls) == 0:
            print("✗ No data found")
            return False
        
        # Use only first few samples
        n_samples = min(max_samples, len(ds_cls))
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(ds_cls, indices)
        
        loader = DataLoader(
            subset, batch_size=min(2, n_samples), shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        print(f"  Testing on {n_samples} samples...")
        summary, df, y_true, y_prob = eval_classifier(
            model, loader, device, 
            threshold=0.5, 
            class_names=config["CLASS_NAMES"],
            show_pbar=True
        )
        
        print(f"✓ Classification eval successful")
        print(f"  - Loss: {summary['loss']:.4f}")
        print(f"  - mAP (macro): {summary['map_macro']:.4f}")
        print(f"  - AUC (macro): {summary['auc_macro']:.4f}")
        print(f"  - F1 (macro): {summary['f1_macro']:.4f}")
        print(f"  - F1 (micro): {summary['f1_micro']:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Classification eval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pseudo_miou(model, device, config, preprocess, max_samples=5):
    """Test pseudo mIoU evaluation on small subset"""
    print("\n" + "="*60)
    print("TEST: Pseudo mIoU Evaluation")
    print("="*60)
    
    val_img_dir = config.get("VAL_IMG_DIR", None)
    val_mask_dir = config.get("VAL_MASK_DIR", None)
    
    if not val_img_dir or not val_mask_dir:
        print("✗ VAL_IMG_DIR or VAL_MASK_DIR not set")
        return False
    
    if not os.path.isdir(val_img_dir) or not os.path.isdir(val_mask_dir):
        print("✗ Validation directories not found")
        return False
    
    target_size = int(config["TARGET_SIZE"])
    eval_config = config.get("EVAL", {})
    
    try:
        # Create small dataset
        ds = BCSSValDataset(val_img_dir, val_mask_dir, target_size=target_size)
        
        if len(ds) == 0:
            print("✗ No validation data found")
            return False
        
        # Limit to first few samples
        n_samples = min(max_samples, len(ds))
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(ds, indices)
        
        # Create custom loader with small batch
        loader = DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        print(f"  Testing on {n_samples} samples...")
        
        # Override eval_config batch size for test
        eval_config_test = eval_config.copy()
        eval_config_test["BATCH_SIZE"] = 1
        eval_config_test["NUM_WORKERS"] = 0
        
        # We need to manually run pseudo_miou logic on subset
        # For quick test, just check forward pass works
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 2:  # Only test 2 batches
                    break
                imgs = batch["img_tensor"].to(device)
                fused_logits, fused_cam, _, _, extras = model.forward_cam_logits_multiscale(imgs)
                print(f"  Batch {i+1}: logits={fused_logits.shape}, cam={fused_cam.shape}")
        
        print(f"✓ Pseudo mIoU forward pass successful")
        print(f"  (Full pseudo mIoU computation skipped for speed)")
        return True
        
    except Exception as e:
        print(f"✗ Pseudo mIoU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_components(model, device):
    """Test individual model components"""
    print("\n" + "="*60)
    print("TEST: Model Components")
    print("="*60)
    
    try:
        # Test prompt bank
        print("  Testing Prompt Bank...")
        coarse, fine, idx_by_class = model.prompt_bank(model.conch, device)
        print(f"    ✓ Coarse embeddings: {coarse.shape}")
        print(f"    ✓ Fine embeddings: {fine.shape}")
        print(f"    ✓ Index by class: {len(idx_by_class)} classes")
        
        # Test knowledge bank
        print("  Testing Knowledge Bank...")
        know_tok = model.knowledge_bank.get_knowledge_tokens_vision()
        know_text = model.knowledge_bank.get_knowledge_for_labels_textspace()
        print(f"    ✓ Knowledge tokens (vision): {know_tok.shape}")
        print(f"    ✓ Knowledge text: {know_text.shape}")
        
        # Test image prototypes (if enabled)
        if getattr(model, "use_img_protos", False):
            print("  Testing Image Prototypes...")
            print(f"    ✓ Prototypes shape: {model.img_prototypes.shape}")
            print(f"    ✓ Proto projection: {model.proto_proj}")
        
        print("✓ All components working")
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_test(config_path="config.yaml", quick=True):
    """Main test function"""
    print("\n" + "="*60)
    print("BCSS-WSSS MODEL QUICK TEST")
    print("="*60)
    
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["SEED"])
    
    device = config["DEVICE"]
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    if not use_cuda:
        device = "cpu"
        print("Warning: CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    
    # Load CONCH
    print("\nLoading CONCH model...")
    conch_model, preprocess = create_model_from_pretrained(
        config["CONCH_ARCH"], 
        config["CONCH_HF"], 
        hf_auth_token=config.get("HF_TOKEN", None),
    )
    conch_model = conch_model.to(device)
    conch_tokenizer = get_tokenizer()
    
    # Inject LoRA
    if config.get("ENABLE_TEXT_LORA", True):
        n_replaced = inject_lora_conch_text(
            conch_model.text,
            r=config.get("LORA_R", 8),
            alpha=config.get("LORA_ALPHA", 16),
            dropout=config.get("LORA_DROPOUT", 0.05),
            mode=config.get("LORA_TEXT_TARGET_MODE", "attn_mlp"),
        )
        print(f"Injected LoRA into {n_replaced} layers")
        conch_lora_sanity_check(conch_model)
    
    # Load prototypes (if enabled)
    prototypes = None
    if config.get("USE_IMAGE_PROTOS", False):
        prototype_path = config.get("PROTOTYPE_PATH", "")
        if prototype_path and os.path.exists(prototype_path):
            print(f"Loading prototypes from {prototype_path}...")
            prototypes = load_prototypes_cluster_bank(
                prototype_path, device=device, 
                use_all=config.get("PROTO_USE_ALL_BANK", True)
            )
            if prototypes is not None:
                print(f"  Loaded prototypes: {prototypes.shape}")
        else:
            print("Warning: USE_IMAGE_PROTOS=True but prototype_path not found")
    
    # Build model
    print("\nBuilding model...")
    model = Model(
        conch_model=conch_model,
        conch_tokenizer=conch_tokenizer,
        config=config,
        prototypes=prototypes,
        num_classes=4,
    ).to(device)
    
    print("Building text/knowledge bases...")
    model.build_text_knowledge_bases(device)
    
    # Build initial fine cache
    if config.get("FINE_CACHE_ENABLE", True):
        print("Building fine prompt cache...")
        model.prompt_bank.update_fine_cache(
            model.conch,
            device=torch.device(device),
            dtype=config.get("FINE_CACHE_DTYPE", "fp16"),
            chunk=config.get("FINE_CACHE_CHUNK", 64),
        )
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Model components
    results["components"] = test_model_components(model, device)
    
    # Test 2: Forward pass
    results["forward"] = test_model_forward(model, device, batch_size=2, 
                                           target_size=int(config["TARGET_SIZE"]))
    
    # Test 3: TTA
    results["tta"] = test_tta(model, device, batch_size=1, 
                             target_size=int(config["TARGET_SIZE"]))
    
    # Test 4: Classification eval (if data available)
    if not quick:
        results["cls_eval"] = test_classification_eval(model, device, config, preprocess, max_samples=10)
        results["pseudo_miou"] = test_pseudo_miou(model, device, config, preprocess, max_samples=5)
    else:
        print("\n(Skipping data-dependent tests in quick mode)")
        results["cls_eval"] = None
        results["pseudo_miou"] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(v for v in results.values() if v is not None)
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test BCSS-WSSS model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to config file")
    parser.add_argument("--quick", action="store_true", default=True,
                       help="Quick mode (skip data-dependent tests)")
    parser.add_argument("--full", action="store_true", default=False,
                       help="Full test mode (include data-dependent tests)")
    
    args = parser.parse_args()
    quick_mode = not args.full
    
    success = main_test(config_path=args.config, quick=quick_mode)
    sys.exit(0 if success else 1)