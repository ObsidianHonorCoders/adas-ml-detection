import pytest
from pathlib import Path

def extract_sequence_id(filename: str) -> str:
    """Extracts the BDD100K sequence hash from a filename."""
    return filename.split("-")[0]

def get_split_sequences(split_dir: Path) -> set:
    """Returns a set of unique sequence IDs found in a given directory."""
    if not split_dir.exists():
        return set()
    return {extract_sequence_id(p.name) for p in split_dir.glob("*.jpg")}

def test_no_sequence_leakage_between_splits():
    """Ensures train, val, and test splits share absolutely no video sequences."""
    processed_dir = Path("data/processed/images")
    
    train_seqs = get_split_sequences(processed_dir / "train")
    val_seqs = get_split_sequences(processed_dir / "val")
    test_seqs = get_split_sequences(processed_dir / "test")
    
    # If the directories are empty (e.g., in CI before data generation), skip the test
    if not train_seqs and not test_seqs:
        pytest.skip("No processed data found to test.")
        
    assert train_seqs.isdisjoint(test_seqs), "CRITICAL: Train and Test sets share overlapping sequence IDs!"
    assert train_seqs.isdisjoint(val_seqs), "CRITICAL: Train and Val sets share overlapping sequence IDs!"
    assert val_seqs.isdisjoint(test_seqs), "CRITICAL: Val and Test sets share overlapping sequence IDs!"
