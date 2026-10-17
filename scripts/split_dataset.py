import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

def split_bdd_sequences(raw_images_dir: Path, raw_labels_dir: Path, output_dir: Path, seed: int = 42):
    """Splits BDD100K frames into train/val/test ensuring no sequence leakage."""
    random.seed(seed)
    
    # 1. Group images by their 17-character sequence ID
    sequences = defaultdict(list)
    for img_path in raw_images_dir.glob("*.jpg"):
        # BDD100K format: b1c662ae-00000000.jpg -> b1c662ae
        seq_id = img_path.stem.split("-")[0]
        sequences[seq_id].append(img_path)
        
    seq_ids = list(sequences.keys())
    seq_ids.sort() # Ensure deterministic order before shuffle
    random.shuffle(seq_ids)
    
    # 2. Calculate split indices (70% train, 15% val, 15% test)
    n = len(seq_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    splits = {
        "train": set(seq_ids[:n_train]),
        "val": set(seq_ids[n_train:n_train + n_val]),
        "test": set(seq_ids[n_train + n_val:])
    }
    
    # Assert zero leakage
    assert splits["train"].isdisjoint(splits["test"]), "Leakage: Train and Test share sequences!"
    assert splits["train"].isdisjoint(splits["val"]), "Leakage: Train and Val share sequences!"
    
    # 3. Copy files to YOLO structure
    for split_name, assigned_seqs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        frames_copied = 0
        for seq_id in assigned_seqs:
            for img_path in sequences[seq_id]:
                # Copy image
                shutil.copy2(img_path, img_out / img_path.name)
                
                # Copy corresponding label if it exists
                label_path = raw_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, lbl_out / label_path.name)
                
                frames_copied += 1
                
        print(f"Split '{split_name}': {len(assigned_seqs)} sequences, {frames_copied} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/interim/images", type=str)
    parser.add_argument("--labels", default="data/interim/labels", type=str)
    parser.add_argument("--output", default="data/processed", type=str)
    args = parser.parse_args()
    
    split_bdd_sequences(Path(args.images), Path(args.labels), Path(args.output))
