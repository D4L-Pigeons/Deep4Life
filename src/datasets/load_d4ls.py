import os
import anndata
# import pyometiff
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH: Path = Path(__file__).parent.parent.parent / "data"
<<<<<<< HEAD
=======
ORIGINAL_IMAGE_DATA_SUBDIR: str = "images_masks"
ORIGINAL_IMAGES_SUBDIR: str = "img"
ORIGINAL_MASKS_SUBDIR: str = "masks"
ANNDATA_FILENAME = "cell_data.h5ad"

TRAIN_DATA_PATH: Path = DATA_PATH / "train"
TRAIN_ANNDATA_PATH: Path = TRAIN_DATA_PATH / ANNDATA_FILENAME
TRAIN_IMAGE_DATA_DIR: Path = TRAIN_DATA_PATH / ORIGINAL_IMAGE_DATA_SUBDIR
TRAIN_IMAGE_DATA_IMAGES: Path = TRAIN_IMAGE_DATA_DIR / ORIGINAL_IMAGES_SUBDIR
TRAIN_IMAGE_DATA_MASKS: Path = TRAIN_IMAGE_DATA_DIR / ORIGINAL_MASKS_SUBDIR
>>>>>>> master

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

<<<<<<< HEAD
# we are not using image data in this project
# HOW TO GET IMAGE NAMES?
# image_names = train_anndata.obs["image"].values()
# todo: https://bodenmillergroup.github.io/ImcSegmentationPipeline/prepro.html#conversion-from-ometiff-to-multi-channel-tiffs
    
# def remove_heavy_tail(images):
#     return np.arcsinh(image_array[2].flatten() / 5.0)


# def load_image(image_name: str) -> np.ndarray:
#     r"""
#     Load an image from the Deep4Life dataset.
    
#     Input:
#     - image_name (str): The name of the image to load.
    
#     Output:
#     - np.ndarray: The image.
#     """
    
#     image_path = TRAIN_IMAGE_DATA_IMAGES / image_name
#     image_reader = pyometiff.OMETIFFReader(fpath=image_path)
#     image_array, _, _ = image_reader.read()
    
#     return image_array


# def load_mask(image_name: str) -> np.ndarray:
#     r"""
#     Load a mask from the Deep4Life dataset.
    
#     Input:
#     - image_name (str): The name of the mask to load.
    
#     Output:
#     - np.ndarray: The mask.
#     """
    
#     mask_path = TRAIN_IMAGE_DATA_MASKS / image_name
#     mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)
#     mask_array, _, _ = mask_reader.read()
    
#     return mask_array
=======

# HOW TO GET IMAGE NAMES?
# image_names = train_anndata.obs["image"].values()


def remove_heavy_tail(images):
    return np.arcsinh(image_array[2].flatten() / 5.0)


def load_image(image_name: str) -> np.ndarray:
    r"""
    Load an image from the Deep4Life dataset.

    Input:
    - image_name (str): The name of the image to load.

    Output:
    - np.ndarray: The image.
    """

    image_path = TRAIN_IMAGE_DATA_IMAGES / image_name
    image_reader = pyometiff.OMETIFFReader(fpath=image_path)
    image_array, _, _ = image_reader.read()

    return image_array


def load_mask(image_name: str) -> np.ndarray:
    r"""
    Load a mask from the Deep4Life dataset.

    Input:
    - image_name (str): The name of the mask to load.

    Output:
    - np.ndarray: The mask.
    """

    mask_path = TRAIN_IMAGE_DATA_MASKS / image_name
    mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)
    mask_array, _, _ = mask_reader.read()

    return mask_array
>>>>>>> master
