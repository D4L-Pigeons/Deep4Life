from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler


class XGBoostModel:
    def __init__(self, args):
        best_xgb_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.3,
            "objective": 'multi:softmax'
        }
        self.xgboost = XGBClassifier(**best_xgb_params)
        self.scaler = MinMaxScaler()

    def train(self, X_train, y_train):
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.xgboost.fit(X_train_scaled, y_train)

    def pred(self, X):
        X_scaled = self.scaler.transform(X)

        return self.xgboost.pred(X_scaled)