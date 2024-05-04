import os
import anndata
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"

TRAIN_DATA_PATH: Path = DATA_PATH / "train"
TRAIN_ORIGINAL_IMAGE_DATA_SUBDIR: str = 'images_masks'
TRAIN_ORIGINAL_IMAGES_SUBDIR: str = "img"
TRAIN_ORIGINAL_MASKS_SUBDIR: str = "masks"
TRAIN_ANNDATA_FILENAME = "cell_data.h5ad"
TRAIN_ANNDATA_PATH: Path = TRAIN_DATA_PATH / TRAIN_ANNDATA_FILENAME
TRAIN_IMAGE_DATA_DIR: Path = TRAIN_DATA_PATH / TRAIN_ORIGINAL_IMAGE_DATA_SUBDIR
TRAIN_IMAGE_DATA_IMAGES: Path = TRAIN_IMAGE_DATA_DIR / TRAIN_ORIGINAL_IMAGES_SUBDIR
TRAIN_IMAGE_DATA_MASKS: Path = TRAIN_IMAGE_DATA_DIR / TRAIN_ORIGINAL_MASKS_SUBDIR

TEST_DATA_PATH: Path = DATA_PATH / "test"
TEST_ORIGINAL_IMAGE_DATA_SUBDIR: str = 'images_masks'
TEST_ORIGINAL_IMAGES_SUBDIR: str = "img"
TEST_ORIGINAL_MASKS_SUBDIR: str = "masks"
TEST_ANNDATA_FILENAME = "cell_data.h5ad"
TEST_ANNDATA_PATH: Path = TEST_DATA_PATH / TEST_ANNDATA_FILENAME
TEST_IMAGE_DATA_DIR: Path = TEST_DATA_PATH / TEST_ORIGINAL_IMAGE_DATA_SUBDIR
TEST_IMAGE_DATA_IMAGES: Path = TEST_IMAGE_DATA_DIR / TEST_ORIGINAL_IMAGES_SUBDIR
TEST_IMAGE_DATA_MASKS: Path = TEST_IMAGE_DATA_DIR / TEST_ORIGINAL_MASKS_SUBDIR


def load_full_anndata() -> anndata.AnnData:
    r"""
    Load the full anndata object from the Deep4Life dataset.

    Output:
    - anndata.AnnData: The full anndata object.
    """
    
    train_anndata = anndata.read_h5ad(TRAIN_ANNDATA_PATH)

    return train_anndata