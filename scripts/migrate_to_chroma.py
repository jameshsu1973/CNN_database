#!/usr/bin/env python3
"""
Migration script to convert pickle-based vaults to ChromaDB
Usage: python scripts/migrate_to_chroma.py [vault_path] [chroma_path]
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.config import get_config


def migrate_vault(pickle_path: str, chroma_path: str = None):
    """Migrate a pickle vault to ChromaDB."""
    config = get_config()

    if chroma_path is None:
        chroma_path = config.chroma_persist_dir

    print(f"Migrating {pickle_path} to ChromaDB at {chroma_path}")

    # Create ChromaDB vault
    vault = ChromaWeightVault(persist_directory=chroma_path)

    # Migrate data
    count = vault.migrate_from_pickle(pickle_path)

    # Save
    vault.save_vault()

    print(f"Migration complete! Migrated {count} entries.")
    return count


def migrate_all():
    """Migrate all default vaults."""
    config = get_config()

    vaults = [
        ("./vault/vault.pkl", "./chroma_db"),
        ("./detection_vault/detection_vault.pkl", "./chroma_db_detection"),
        ("./cat_detection_vault/detection_vault.pkl", "./chroma_db_cat"),
        ("./resnet_detection_vault/detection_vault.pkl", "./chroma_db_resnet"),
    ]

    total = 0
    for pickle_path, chroma_path in vaults:
        if os.path.exists(pickle_path):
            print(f"\n{'='*50}")
            print(f"Processing: {pickle_path}")
            print(f"{'='*50}")
            try:
                count = migrate_vault(pickle_path, chroma_path)
                total += count
            except Exception as e:
                print(f"Error migrating {pickle_path}: {e}")
        else:
            print(f"Skipping {pickle_path} (not found)")

    print(f"\n{'='*50}")
    print(f"Total entries migrated: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No args - migrate all default vaults
        migrate_all()
    elif len(sys.argv) >= 2:
        # Migrate specific vault
        pickle_path = sys.argv[1]
        chroma_path = sys.argv[2] if len(sys.argv) > 2 else None
        migrate_vault(pickle_path, chroma_path)
    else:
        print("Usage: python migrate_to_chroma.py [vault.pkl] [chroma_path]")
        print("       python migrate_to_chroma.py  # migrate all default vaults")
