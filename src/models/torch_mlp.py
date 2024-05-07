import time

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.ModelBase import ModelBase

pd.set_option('display.max_columns', None)

class MLP(nn.Module):
    def __init__(self):
      super().__init__()
      self.layers = nn.Sequential(
        nn.Linear(40, 40*4),
        nn.ReLU(),
        nn.Linear(40*4, 40*4),
        nn.ReLU(),
        nn.Linear(40*4, 14),
      )
    def forward(self, input):
       return self.layers(input)

class TorchMLP(ModelBase):
    def __init__(self, config):
        self.mlp = MLP()

    def train(self, data: anndata.AnnData) -> None:
        X_train = data.layers['exprs']
        y_train = data.obs['cell_labels'].cat.codes.to_numpy()
        
        self._train(X_train, y_train, log=True, early_stopping=True)
    
    def predict(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers['exprs']
        X = torch.tensor(X).float()
        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(X)
        preds = torch.argmax(logits, dim=1).detach().numpy()
        pred_labels = data.obs["cell_labels"].cat.categories[preds].to_numpy()
        
        return pred_labels
    
    def predict_proba(self, data: anndata.AnnData) -> np.ndarray:
        X = data.layers['exprs']
        X = torch.tensor(X).float()
        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(X)
        prediction_probabilities = nn.functional.softmax(logits, dim=1).detach().numpy()
        
        return prediction_probabilities

    def save(self, file_path: str) -> None:
        torch.save(self.mlp.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        self.mlp.load_state_dict(torch.load(file_path))
    
    def _train(self, X_train, y_train, log=False, early_stopping=False):
        # train config. Probably it should be moved to config directory
        BATCH_SIZE = 200
        LR = 0.001
        WEIGHT_DECAY = 0.0001
        MAX_EPOCHS = 40
        DEVICE = torch.device('cpu')
        PRINT_EVERY = 10
        TOLERANCE = 0.0001
        N_INTER_NO_CHANGE = 10

        if early_stopping:
            X_train_to_loader, X_val, y_train_to_loader, y_val = train_test_split(torch.tensor(X_train).float(),
                                                              torch.tensor(y_train).long(),
                                                              test_size=0.1,
                                                              random_state=42)
            train_data_loader = DataLoader(list(zip(X_train_to_loader, y_train_to_loader)), shuffle=True, batch_size=BATCH_SIZE)
        else:
            train_data_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=BATCH_SIZE)

        self.mlp = MLP()
        optimizer = optim.Adam(self.mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        self.mlp.train()
        start = time.time()

        self.tol = TOLERANCE
        self._no_improvement_count = 0
        self.best_validation_accuracy = 0
        self.validation_accuracies = []
        self.train_losses = []
        self.best_loss = 100.
        self.best_model_weights = None

        for epoch in range(MAX_EPOCHS):

            if log:
                print("\r   %dm: epoch %d [%s] %d%%  loss = %s" %\
                ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='')

            accumulated_loss = 0.0
            for i, (input, targets) in enumerate(train_data_loader):
                input, targets = input.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                pred = self.mlp(input)
                loss = criterion(pred, targets)

                loss.backward()
                optimizer.step()

                accumulated_loss += loss.item() 

                if log and (i + 1) % PRINT_EVERY == 0:
                        p = int(100 * (i + 1) / len(train_data_loader))
                        avg_loss = accumulated_loss / (i+1)

                        print("\r   %dm: epoch %d [%s%s] %d%% train loss = %.10f" %\
                        ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='')

                        self.train_losses.append(avg_loss)

            avg_train_loss = accumulated_loss / len(train_data_loader)
            self.train_losses.append(avg_train_loss)

            self._update_no_improvement_count(early_stopping, X_val, y_val, log)

            if self._no_improvement_count > N_INTER_NO_CHANGE:
                break	

        if early_stopping:
            self.mlp.load_state_dict(self.best_model_weights)
    
    def _update_no_improvement_count(self, early_stopping, X_val, y_val, log):
        if early_stopping:
            self.validation_accuracies.append(self._compute_accuracy(X_val, y_val))
            self.mlp.train()

            if log:
                print(" Validation score: %f" % self.validation_accuracies[-1])

            last_valid_score = self.validation_accuracies[-1]

            if last_valid_score < (self.best_validation_accuracy + self.tol):
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0

            if last_valid_score > self.best_validation_accuracy:
                self.best_validation_accuracy = last_valid_score
                self.best_model_weights = self.mlp.state_dict().copy()
        else:
            if self.train_losses[-1] > self.best_loss - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.train_losses[-1] < self.best_loss:
                self.best_loss = self.train_losses[-1]
    
    def _compute_accuracy(self, X_val, y_val):
        self.mlp.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to compute gradients during validation
            pred = self.mlp(X_val)
            correct = (torch.argmax(pred, dim=1) == y_val).float().sum()
            accuracy = correct / len(y_val)

        return accuracy