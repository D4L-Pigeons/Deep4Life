import anndata
import numpy as np


class ModelBase:

    def train(self, data: anndata.AnnData) -> None:
        pass

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        raise NotImplementedError()

    def save(self, file_path: str) -> None:
        pass

    def load(self, file_path: str) -> None:
        pass