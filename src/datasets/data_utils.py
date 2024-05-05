import os
import anndata
import pyometiff
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
ORIGINAL_IMAGE_DATA_SUBDIR: str = 'images_masks'
ORIGINAL_IMAGES_SUBDIR: str = "img"
ORIGINAL_MASKS_SUBDIR: str = "masks"
ANNDATA_FILENAME = "cell_data.h5ad"
    
TRAIN_DATA_PATH: Path = DATA_PATH / "train"
TRAIN_ANNDATA_PATH: Path = TRAIN_DATA_PATH / ANNDATA_FILENAME
TRAIN_IMAGE_DATA_DIR: Path = TRAIN_DATA_PATH / ORIGINAL_IMAGE_DATA_SUBDIR
TRAIN_IMAGE_DATA_IMAGES: Path = TRAIN_IMAGE_DATA_DIR / ORIGINAL_IMAGES_SUBDIR
TRAIN_IMAGE_DATA_MASKS: Path = TRAIN_IMAGE_DATA_DIR / ORIGINAL_MASKS_SUBDIR


def load_full_anndata() -> anndata.AnnData:
    r"""
    Load the full anndata object from the Deep4Life dataset.

    Output:
    - anndata.AnnData: The full anndata object.
    """
    
    train_anndata = anndata.read_h5ad(TRAIN_ANNDATA_PATH)

    return train_anndata