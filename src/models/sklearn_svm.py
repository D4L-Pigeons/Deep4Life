import anndata
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from models.ModelBase import ModelBase
import abc
from joblib import dump, load


class SVMSklearnModel(ModelBase, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.svm = self.initialize_svm(config)
        self.scaler = MinMaxScaler()

    @abc.abstractmethod
    def initialize_svm(self, config):
        pass

    def train(self, data: anndata.AnnData) -> None:
        X_train = data.layers["exprs"]
        y_train = data.obs["cell_labels"].cat.codes.tolist()

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.svm.fit(X_train_scaled, y_train)

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers["exprs"]
        X_scaled = self.scaler.transform(X)

        prediction = self.svm.predict(X_scaled)

        return data.obs["cell_labels"].cat.categories[prediction].to_numpy()

    def predict_proba(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers["exprs"]
        X_scaled = self.scaler.transform(X)

        prediction_probabilities = self.svm.predict_proba(X_scaled)
        return prediction_probabilities

    def save(self, file_path: str) -> str:
        path_with_ext = file_path + ".joblib"
        dump(self.svm, path_with_ext)
        return path_with_ext

    def load(self, file_path: str) -> None:
        self.svm = load(file_path)


class SVMSklearnSVC(SVMSklearnModel):
    def __init__(self, config):
        super().__init__(config)

    def initialize_svm(self, config):
        return SVC(**config)


class SVMSklearnLinearSVC(SVMSklearnModel):
    def __init__(self, config):
        super().__init__(config)

    def initialize_svm(self, config):
        return LinearSVC(**config)


class SVMSklearnNuSVC(SVMSklearnModel):
    def __init__(self, config):
        super().__init__(config)

    def initialize_svm(self, config):
        return NuSVC(**config)
