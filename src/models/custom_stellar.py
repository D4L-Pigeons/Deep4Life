from models.vanilla_stellar import VanillaStellarEncoder, VanillaStellarClassifficationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from models.ModelBase import ModelBase
from datasets.stellar_data import StellarDataloader, make_graph_list_from_anndata
from utils import calculate_batch_accuracy
import anndata
import pandas as pd


class CustomStellarEncoder_1(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.

    Adds the BatchNorm after each layer of the VanillaStellarEncoder
    """
    def __init__(self,
                 input_dim: int,
                 hid_dim: int=128
                 ):
        super(CustomStellarEncoder_1, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim))
        self.graph_conv = SAGEConv(hid_dim, hid_dim)
        self.graph_conv_bn = nn.BatchNorm1d(hid_dim)
            

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        out_feat = self.graph_conv(feat, edge_index)
        out_feat = self.graph_conv_bn(out_feat)
        return feat, out_feat


class CustomStellarEncoder_2(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.
    
    Doubles each layer of the CustomStellalrEncoder_0
    """
    def __init__(self,
                 input_dim: int,
                 hid_dim: int=128
                 ):
        super(CustomStellarEncoder_2, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            )
        self.hidden_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            )
        self.graph_conv1 = SAGEConv(hid_dim, hid_dim)
        self.graph_conv1_bn = nn.BatchNorm1d(hid_dim)
        self.graph_conv2 = SAGEConv(hid_dim, hid_dim)
        self.graph_conv2_bn = nn.BatchNorm1d(hid_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        feat = F.relu(self.hidden_linear(feat))
        out_feat = self.graph_conv1(feat, edge_index)
        out_feat = self.graph_conv1_bn(out_feat)
        out_feat = self.graph_conv2(out_feat, edge_index)
        out_feat = self.graph_conv2_bn(out_feat)
        return feat, out_feat


ENCODER_IMPLEMENTATIONS = [
    VanillaStellarEncoder,
    CustomStellarEncoder_1,
    CustomStellarEncoder_2
]


r"""
I replace the VanillaStellarNormedLinear from the paper with the LayerNorm, which is not strictly equivalent in computations
"""
class CustomStellarClassifficationHead_1(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    Replaces the VanillaStellarNormedLinear with LayerNorm
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 temperature: float
                 ):
        super(CustomStellarClassifficationHead_1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.LayerNorm(num_classes)
        )
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out * self.temperature

class CustomStellarClassifficationHead_1(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    Replaces the VanillaStellarNormedLinear with LayerNorm
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 temperature: float
                 ):
        super(CustomStellarClassifficationHead_1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.LayerNorm(num_classes)
        )
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out * self.temperature


CLASSIFICATION_HEAD_IMPLEMENTATIONS = [
    VanillaStellarClassifficationHead,
    CustomStellarClassifficationHead_1
]


class CustomStellarModel(nn.Module):
    r"""
    A model that uses a graph encoder followed by a classification head to make predictions.

    Allows for flexibly altering the VanillaStellarReduced
    """
    def __init__(self,
                 encoder_constructor: nn.Module,
                 fc_net_constructor: nn.Module,
                 input_dim: int,
                 hid_dim: int,
                 num_classes: int,
                 temperature: float=10.0
                 ):
        super(CustomStellarModel, self).__init__()
        self.encoder = encoder_constructor(input_dim, hid_dim)
        self.fc_net = fc_net_constructor(hid_dim, num_classes, temperature=temperature)

    def forward(self, data: Data):
        _, out_feat = self.encoder(data)
        assert type(out_feat) == Tensor, f"Expected Tensor, got {type(out_feat)}"
        out = self.fc_net(out_feat)
        return out, out_feat
    

class CustomStellarReduced(ModelBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = CustomStellarModel(
            ENCODER_IMPLEMENTATIONS[cfg.encoder_impl],
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[cfg.classification_head_impl],
            cfg.input_dim,
            cfg.hid_dim,
            cfg.num_classes
            ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.label_encoder = LabelEncoder().fit(cfg.target_labels)

    def train(self, data: anndata.AnnData) -> None:
        self.model = CustomStellarModel(
            ENCODER_IMPLEMENTATIONS[self.cfg.encoder_impl],
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[self.cfg.classification_head_impl],
            self.cfg.input_dim,
            self.cfg.hid_dim,
            self.cfg.num_classes
            ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        graphs = make_graph_list_from_anndata(data, self.label_encoder, self.cfg.distance_threshold)
        train_data_loader = StellarDataloader(graphs, batch_size=self.cfg.batch_size)
        
        self._train(train_data_loader, self.cfg.epochs)

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        graphs = make_graph_list_from_anndata(data, self.label_encoder, self.cfg.distance_threshold)
        batched_graphs =  Batch.from_data_list(graphs)
        cell_ids = np.concatenate(batched_graphs.cell_ids)
        
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batched_graphs)
        preds = logits.argmax(dim=1).detach().numpy()
        pred_labels = data.obs["cell_labels"].cat.categories[preds]
        
        return pd.Series(data=pred_labels, index=cell_ids).reindex(data.obs.index).to_numpy()

    def save(self, file_path: str) -> None:
        raise NotImplementedError()

    def load(self, file_path: str) -> None:
        raise NotImplementedError()

    def _train(
            self,
            train_loader: DataLoader,
            epochs: int,
            valid_loader: DataLoader | None = None,
            return_valid_acc: bool = False
            ) -> Optional[float]:
        r"""
        Trains the model in a supervised manner.
        """
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            sum_loss = 0.0
            acc_sum = 0.0
            self.model.train()
            train_progress_bar = tqdm(train_loader, desc=f"Training - epoch {epoch}", leave=True)
            for batch_number, batch in enumerate(train_progress_bar):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(batch)
                loss = cross_entropy_loss_fn(output, batch.y)
                sum_loss += loss.item()
                acc_sum += calculate_batch_accuracy(output, batch.y)
                loss.backward()
                self.optimizer.step()
                train_progress_bar.set_postfix({"Loss": sum_loss / (batch_number + 1), "Accuracy": acc_sum / (batch_number + 1)})
            
            if valid_loader != None:
                val_loss_sum = 0
                val_acc_sum = 0
                self.model.eval()
                val_progress_bar = tqdm(valid_loader, desc=f"Validation - epoch {epoch}", leave=True)
                for batch_number, batch in enumerate(val_progress_bar):
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        output, _ = self.model(batch)
                    loss = F.cross_entropy(output, batch.y)
                    val_loss_sum += loss.item()
                    val_acc_sum += calculate_batch_accuracy(output, batch.y)
                    val_progress_bar.set_postfix({"Loss": val_loss_sum / (batch_number + 1), "Accuracy": val_acc_sum / (batch_number + 1)})
        
        if return_valid_acc:
            return val_acc_sum / (batch_number + 1)