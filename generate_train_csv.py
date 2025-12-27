"""
Script to generate train.csv from image filenames
Usage: python generate_train_csv.py
"""
import os
import pandas as pd

# Paths - adjust these if needed
DATA_ROOT = "/home/s12gbn4/aivn2026-research/thaivan/BCSS-WSSS/training"
OUTPUT_CSV = "/home/s12gbn4/aivn2026-research/thaivan/BCSS-WSSS/train.csv"

def generate_csv_from_filenames(data_root, output_path):    
    """
    Generate CSV file from image filenames with label encoding
    
    Format: filename[abcd].png where:
    - a = Tumor (0 or 1)
    - b = Stroma (0 or 1)
    - c = Lymphocytic infiltrate (0 or 1)
    - d = Necrosis (0 or 1)
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    files = [f for f in os.listdir(data_root) if f.endswith('.png')]
    data = []
    
    print(f"Creating labels for {len(files)} images...")
    
    for filename in files:
        # Parse label from filename: filename[abcd].png
        try:
            label_str = filename.rsplit('[', 1)[1].rsplit(']', 1)[0]
            
            if len(label_str) != 4:
                print(f"Skipping strange file: {filename}")
                continue
                
            # Parse từng ký tự (0 hoặc 1)
            tumor = int(label_str[0])
            stroma = int(label_str[1])
            lymph = int(label_str[2])
            necrosis = int(label_str[3])
            
            data.append({
                'image_id': filename.replace('.png', ''),
                'filename': filename,                     
                'Tumor': tumor,
                'Stroma': stroma,
                'Lymphocytic infiltrate': lymph,
                'Necrosis': necrosis
            })
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    
    df = pd.DataFrame(data)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved labels to: {output_path}")
    print(f"Total rows: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLabel distribution:")
    print(df[['Tumor', 'Stroma', 'Lymphocytic infiltrate', 'Necrosis']].sum())
    
    return df

if __name__ == "__main__":
    generate_csv_from_filenames(DATA_ROOT, OUTPUT_CSV)

