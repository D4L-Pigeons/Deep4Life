from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import anndata
import numpy as np

from models.ModelBase import ModelBase

class SklearnMLP(ModelBase):
    def __init__(self, args):
        self.mlp_classifier = MLPClassifier(solver='adam',
                                            alpha=1e-5,
                                            hidden_layer_sizes=(40*4, 40*4),
                                            random_state=1,
                                            early_stopping=True,
                                            verbose=True)

    def train(self, data: anndata.AnnData) -> None:
        X_train = data.layers['exprs']
        y_train = data.obs['cell_labels'].cat.codes.to_numpy()
        
        self.mlp_classifier.fit(X_train, y_train)

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers['exprs']

        prediction = self.mlp_classifier.predict(X)

        return data.obs["cell_labels"].cat.categories[prediction].to_numpy()
    
    def save(self, file_path: str) -> None:
        raise NotImplementedError()

    def load(self, file_path: str) -> None:
        raise NotImplementedError()
