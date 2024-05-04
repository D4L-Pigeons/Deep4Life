from sklearn.neural_network import MLPClassifier

class SklearnMLP:
    def __init__(self, args):
        self.mlp_classifier = MLPClassifier(solver='adam',
                                            alpha=1e-5,
                                            hidden_layer_sizes=(40*4, 40*4),
                                            random_state=1,
                                            early_stopping=True,
                                            verbose=True)

    def train(self, X_train, y_train):
        self.mlp_classifier.fit(X_train, y_train)

    def pred(self, X):
        return self.mlp_classifier.pred(X)