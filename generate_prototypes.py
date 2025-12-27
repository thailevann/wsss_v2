"""
Script to generate prototype features from training images using CONCH
Usage: python generate_prototypes.py [--k_list 3 3 3 3] [--proto_dir ./prototypes]
"""
import os
import sys
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
except ImportError:
    raise ImportError("CONCH not found. Install with: pip install git+https://github.com/Mahmoodlab/CONCH.git")

from utils import set_seed


class CosineSimilarityKMeans:
    """K-means clustering using cosine similarity"""
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Initialize centers randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # Compute similarities
            similarities = cosine_similarity(X, self.cluster_centers_)
            new_labels = np.argmax(similarities, axis=1)
            
            # Update centers
            old_centers = self.cluster_centers_.copy()
            for i in range(self.n_clusters):
                cluster_samples = X[new_labels == i]
                if len(cluster_samples) > 0:
                    # Mean and normalize
                    self.cluster_centers_[i] = cluster_samples.mean(axis=0)
                    norm = np.linalg.norm(self.cluster_centers_[i])
                    if norm > 0:
                        self.cluster_centers_[i] /= norm
            
            # Check convergence
            if np.allclose(old_centers, self.cluster_centers_, atol=1e-6):
                break
                
        return new_labels, similarities, torch.from_numpy(self.cluster_centers_).float()


def extract_features_from_training(
    image_dir,
    label_csv,
    conch_model,
    device,
    target_size=448,
    batch_size=32
):
    """
    Extract features from training images grouped by class
    Only uses images with exactly 1 class (single-label) for prototype generation
    
    Args:
        image_dir: Directory containing training images
        label_csv: Path to CSV with labels
        conch_model: CONCH model (already on device)
        device: Device to use
        target_size: Image size for preprocessing
        batch_size: Batch size for feature extraction
        
    Returns:
        dict: {class_idx: list of features}
    """
    import pandas as pd
    
    # Load labels
    df = pd.read_csv(label_csv)
    
    # Group images by class - only use images with exactly 1 class (single-label)
    class_features = {0: [], 1: [], 2: [], 3: []}  # Tumor, Stroma, Lymph, Necrosis
    class_names = ['Tumor', 'Stroma', 'Lymphocytic infiltrate', 'Necrosis']
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    
    conch_model.eval()
    
    # Pre-filter: only keep images with exactly 1 class (single-label)
    print(f"Filtering images: {len(df)} total images...")
    filtered_df = []
    for idx, row in df.iterrows():
        labels = [int(row[class_names[i]]) for i in range(4)]
        num_classes = sum(labels)
        if num_classes == 1:
            filtered_df.append(row)
    
    print(f"Found {len(filtered_df)} images with exactly 1 class (will process these)")
    print(f"Skipping {len(df) - len(filtered_df)} images with 2-4 classes\n")
    
    if len(filtered_df) == 0:
        print("Warning: No images with exactly 1 class found!")
        return class_features
    
    # Process only filtered images
    for row in tqdm(filtered_df, desc="Extracting features"):
        img_path = os.path.join(image_dir, row['filename'])
        if not os.path.exists(img_path):
            continue
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract features using CONCH encode_image
            with torch.no_grad():
                # Use encode_image which handles output format properly
                # proj_contrast=False returns raw features, normalize=False keeps original scale
                vision_output = conch_model.encode_image(img_tensor, proj_contrast=False, normalize=False)
                
                # Handle tuple output (some CONCH versions return tuple)
                if isinstance(vision_output, (tuple, list)):
                    vision_output = vision_output[0]
                
                # vision_output can be [B, L, D] (tokens) or [B, D] (pooled)
                if vision_output.dim() == 3:
                    # [B, L, D] - use global average pooling over spatial tokens
                    feat = vision_output.mean(dim=1)  # [B, D] - average over tokens
                elif vision_output.dim() == 2:
                    # [B, D] - already pooled
                    feat = vision_output
                elif vision_output.dim() == 4:
                    # [B, C, H, W] - global average pool
                    feat = F.adaptive_avg_pool2d(vision_output, 1).flatten(1)
                else:
                    # Fallback: flatten and pool
                    feat = vision_output.view(vision_output.size(0), -1)
                
                feat = feat.squeeze(0).cpu()  # [D]
                feat = feat / (feat.norm() + 1e-12)  # Normalize
            
            # Add to the single class that this image belongs to (already filtered)
            labels = [int(row[class_names[i]]) for i in range(4)]
            for class_idx in range(4):
                if labels[class_idx] == 1:
                    class_features[class_idx].append(feat)
                    break  # Only one class, so break after finding it
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy arrays
    total_features = 0
    
    for class_idx in range(4):
        if len(class_features[class_idx]) > 0:
            class_features[class_idx] = torch.stack(class_features[class_idx]).numpy()
            total_features += len(class_features[class_idx])
        else:
            class_features[class_idx] = np.array([])
        print(f"Class {class_names[class_idx]}: {len(class_features[class_idx])} samples")
    
    print(f"\nSummary:")
    print(f"  - Total images in dataset: {len(df)}")
    print(f"  - Images with 1 class (processed): {len(filtered_df)}")
    print(f"  - Images skipped (2-4 classes): {len(df) - len(filtered_df)}")
    print(f"  - Total features extracted: {total_features}")
    
    return class_features


def cluster_features_per_class(
    class_features,
    k_list=[3, 3, 3, 3],
    class_order=['Tumor', 'Stroma', 'Lymphocytic infiltrate', 'Necrosis']
):
    """
    Cluster features per class using cosine similarity K-means
    
    Args:
        class_features: dict {class_idx: numpy array [N, D]}
        k_list: List of k values for each class [Tumor, Stroma, Lymph, Necrosis]
        class_order: Order of classes
        
    Returns:
        dict: Prototype data with 'features', 'cumsum_k', 'class_order'
    """
    if len(k_list) != 4:
        raise ValueError("k_list must contain 4 values for [Tumor, Stroma, Lymph, Necrosis]")
    
    all_centers = []
    
    for class_idx, (class_name, k) in enumerate(zip(class_order, k_list)):
        print(f"\n{'='*20} Class: {class_name} (k={k}) {'='*20}")
        
        features = class_features[class_idx]
        
        if len(features) == 0:
            print(f"Warning: No features for {class_name}, using random centers")
            # Create random normalized centers
            if len(all_centers) > 0:
                feat_dim = all_centers[0].shape[-1]
            else:
                feat_dim = 768  # CONCH vision dim
            centers = np.random.randn(k, feat_dim).astype(np.float32)
            centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
            all_centers.append(torch.from_numpy(centers).float())
            continue
        
        if len(features) < k:
            print(f"Warning: Only {len(features)} samples for {class_name}, using all as centers")
            centers = features
            if len(centers.shape) == 1:
                centers = centers.unsqueeze(0)
            all_centers.append(torch.from_numpy(centers).float())
            continue
        
        # Normalize features
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        
        # Cluster
        kmeans = CosineSimilarityKMeans(n_clusters=k, random_state=42)
        cluster_labels, similarities, cluster_centers = kmeans.fit_predict(features_norm)
        
        all_centers.append(cluster_centers)
        
        print(f"Generated {k} cluster centers for {class_name}")
    
    # Concatenate all centers
    all_centers_tensor = torch.cat(all_centers, dim=0)
    
    # Create cumsum_k: [0, k0, k0+k1, k0+k1+k2, k0+k1+k2+k3]
    cumsum_k = [0]
    for k in k_list:
        cumsum_k.append(cumsum_k[-1] + k)
    
    save_info = {
        'features': all_centers_tensor,
        'cumsum_k': cumsum_k,
        'class_order': class_order,
        'k_list': k_list
    }
    
    return save_info


def generate_prototypes(
    image_dir,
    label_csv,
    output_path,
    k_list=[3, 3, 3, 3],
    conch_arch="conch_ViT-B-16",
    conch_hf="hf_hub:MahmoodLab/conch",
    hf_token=None,
    device="cuda",
    target_size=448):
    """
    Main function to generate prototype file
    Only uses images with exactly 1 class (single-label) for prototype generation
    
    Args:
        image_dir: Directory containing training images
        label_csv: Path to CSV with labels
        output_path: Path to save prototype file
        k_list: Number of clusters per class [Tumor, Stroma, Lymph, Necrosis]
        conch_arch: CONCH architecture
        conch_hf: CONCH HuggingFace path
        hf_token: HuggingFace token
        device: Device to use
        target_size: Image size
    """
    set_seed(42)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CONCH model
    print("Loading CONCH model...")
    conch_model, preprocess = create_model_from_pretrained(
        conch_arch,
        conch_hf,
        hf_auth_token=hf_token,
    )
    conch_model = conch_model.to(device)
    conch_model.eval()
    
    # Extract features
    print("\n" + "="*50)
    print("Step 1: Extracting features from training images")
    print("="*50)
    class_features = extract_features_from_training(
        image_dir=image_dir,
        label_csv=label_csv,
        conch_model=conch_model,
        device=device,
        target_size=target_size
    )
    
    # Cluster features
    print("\n" + "="*50)
    print("Step 2: Clustering features per class")
    print("="*50)
    prototype_data = cluster_features_per_class(
        class_features=class_features,
        k_list=k_list
    )
    
    # Save
    print("\n" + "="*50)
    print("Step 3: Saving prototype file")
    print("="*50)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pkl.dump(prototype_data, f)
    
    print(f"\n✓ Prototype file saved to: {output_path}")
    print(f"  Features shape: {prototype_data['features'].shape}")
    print(f"  Cumsum k: {prototype_data['cumsum_k']}")
    print(f"  K list: {prototype_data['k_list']}")
    print("\nClass feature index ranges:")
    for i, class_name in enumerate(prototype_data['class_order']):
        start_idx = prototype_data['cumsum_k'][i]
        end_idx = prototype_data['cumsum_k'][i+1]
        print(f"  {class_name}: {start_idx} to {end_idx-1}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate prototype features from training images")
    parser.add_argument('--image_dir', type=str, 
                       default="/home/s12gbn4/aivn2026-research/thaivan/BCSS-WSSS/training",
                       help='Directory containing training images')
    parser.add_argument('--label_csv', type=str,
                       default="/home/s12gbn4/aivn2026-research/thaivan/BCSS-WSSS/train.csv",
                       help='Path to label CSV file')
    parser.add_argument('--output_path', type=str,
                       default="/home/s12gbn4/aivn2026-research/thaivan/dong/prototypes/bcss_prototypes.pkl",
                       help='Path to save prototype file')
    parser.add_argument('--k_list', type=int, nargs=4, default=[3, 3, 3, 3],
                       help='Number of clusters for each class [Tumor, Stroma, Lymph, Necrosis]')
    parser.add_argument('--conch_arch', type=str, default="conch_ViT-B-16",
                       help='CONCH architecture')
    parser.add_argument('--conch_hf', type=str, default="hf_hub:MahmoodLab/conch",
                       help='CONCH HuggingFace path')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device to use')
    parser.add_argument('--target_size', type=int, default=448,
                       help='Image target size')
    
    args = parser.parse_args()
    
    generate_prototypes(
        image_dir=args.image_dir,
        label_csv=args.label_csv,
        output_path=args.output_path,
        k_list=args.k_list,
        conch_arch=args.conch_arch,
        conch_hf=args.conch_hf,
        hf_token=args.hf_token,
        device=args.device,
        target_size=args.target_size
    )

