import anndata
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPClassifier

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
    
    def predict_proba(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers['exprs']
        
        prediction_probabilities = self.mlp_classifier.predict_proba(X)
        
        return prediction_probabilities
    
    def save(self, file_path: str) -> None:
        dump(self.mlp_classifier, file_path) 

    def load(self, file_path: str) -> None:
        self.mlp_classifier = load(file_path) 
