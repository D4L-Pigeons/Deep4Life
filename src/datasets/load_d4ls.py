import os
import anndata
import pyometiff
import matplotlib.pyplot as plt

import numpy as np


def remove_heavy_tail(images):
    return np.arcsinh(image_array[2].flatten() / 5.0)


# todo: https://bodenmillergroup.github.io/ImcSegmentationPipeline/prepro.html#conversion-from-ometiff-to-multi-channel-tiffs


def load_d4ls_data(TRAIN_DATA_PATH="data/train"):
    ORIGINAL_IMAGE_DATA_SUBDIR = "images_masks"
    ORIGINAL_MASKS_SUBDIR = "masks"
    ORIGINAL_IMAGES_SUBDIR = "img"

    ANNDATA_PATH = "cell_data.h5ad"
    TRAIN_ANNDATA_PATH = os.path.join(TRAIN_DATA_PATH, ANNDATA_PATH)
    TRAIN_IMAGE_DATA_DIR = os.path.join(TRAIN_DATA_PATH, ORIGINAL_IMAGE_DATA_SUBDIR)
    TRAIN_IMAGE_DATA_IMAGES = os.path.join(TRAIN_IMAGE_DATA_DIR, ORIGINAL_IMAGES_SUBDIR)
    TRAIN_IMAGE_DATA_MASKS = os.path.join(TRAIN_IMAGE_DATA_DIR, ORIGINAL_MASKS_SUBDIR)
    train_anndata = anndata.read_h5ad(TRAIN_ANNDATA_PATH)

    image_name = train_anndata.obs.iloc[2]["image"]
    image_path = os.path.join(TRAIN_IMAGE_DATA_IMAGES, image_name)
    mask_path = os.path.join(TRAIN_IMAGE_DATA_MASKS, image_name)
    image_reader = pyometiff.OMETIFFReader(fpath=image_path)
    mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)

    image_array, _, _ = image_reader.read()
    mask_array, _, _ = mask_reader.read()

    pass
