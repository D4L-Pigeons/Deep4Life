import anndata
import numpy as np


class ModelBase:

    def train(self, data: anndata.AnnData) -> None:
        raise NotImplementedError()

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        raise NotImplementedError()

    def predict_proba(self, data: anndata.AnnData) -> np.ndarray:
        raise NotImplementedError()

    def save(self, file_path: str) -> str:
        """
        Returns:
            str: the file path to saved model. This method can add extension to given file_path.
        """
        raise NotImplementedError()

    def load(self, file_path: str) -> None:
        raise NotImplementedError()
