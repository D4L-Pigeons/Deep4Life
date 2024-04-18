from torch.utils.data import Dataset


class CellDataset(Dataset):
    def __init__(self, anndata, image_dir, mask_dir):
        self.anndata = anndata
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.anndata.obs)

    def __getitem__(self, idx):
        # Get information from the dataframe
        image_name = self.anndata.obs.iloc[idx]["image"]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        # Load image and mask using pyometiff
        image_reader = pyometiff.OMETIFFReader(fpath=image_path)
        mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)
        image_array, _, _ = image_reader.read()
        mask_array, _, _ = mask_reader.read()

        cell_info = {
            "sample_id": self.anndata.obs.iloc[idx]["sample_id"],
            "ObjectNumber": self.anndata.obs.iloc[idx]["ObjectNumber"],
            "Pos_X": self.anndata.obs.iloc[idx]["Pos_X"],
            "Pos_Y": self.anndata.obs.iloc[idx]["Pos_Y"],
            # Add other relevant information if needed
        }

        # Preprocess image and mask (optional)
        # You can perform transformations on image and mask here

        return image_array, mask_array, cell_info
