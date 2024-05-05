from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import anndata
import numpy as np

from models.ModelBase import ModelBase


class XGBoostModel(ModelBase):
    def __init__(self, args):
        best_xgb_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.3,
            "objective": 'multi:softmax'
        }
        self.xgboost: XGBClassifier = XGBClassifier(**best_xgb_params)
        self.scaler = MinMaxScaler()

    def train(self, data: anndata.AnnData) -> None:
        X_train = data.layers['exprs']
        y_train = data.obs['cell_labels'].cat.codes.tolist()

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.xgboost.fit(X_train_scaled, y_train)


    def predict(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers['exprs']
        X_scaled = self.scaler.transform(X)

        prediction = self.xgboost.predict(X_scaled)

        return data.obs["cell_labels"].cat.categories[prediction].to_numpy()

    def save(self, file_path: str) -> None:
        raise NotImplementedError()

    def load(self, file_path: str) -> None:
        raise NotImplementedError()

        