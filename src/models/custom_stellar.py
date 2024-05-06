from vanilla_stellar import VanillaStellarNormedLinear, VanillaStellarEncoder, VanillaStellarClassifficationHead
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
# from torch_geometric.loader i/mport DataLoader
# from anndata import AnnData
from typing import Optional, Tuple, Generator
# import numpy as np
from sklearn.preprocessing import LabelEncoder
# import scanpy as sc
from tqdm import tqdm
from models.ModelBase import ModelBase
from datasets.stellar_data import StellarDataloader, make_graph_list_from_anndata
# from utils import calculate_entropy_logits, calculate_entropy_probs, calculate_batch_accuracy, MarginLoss
# from itertools import cycle
# import anndata
# import pandas as pd

class CustomStellarEncoder_0(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.

    Adds the BatchNorm after each layer of the VanillaStellarEncoder
    """
    def __init__(self,
                 input_dim: int,
                 hid_dim: int=128
                 ):
        super(CustomStellarEncoder_0, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim))
        self.graph_conv = nn.Sequential(
            SAGEConv(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim)
            )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        out_feat = self.graph_conv(feat, edge_index)
        return feat, out_feat


class CustomStellarEncoder_1(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.
    
    Doubles each layer of the CustomStellalrEncoder_0
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
            nn.BatchNorm1d(hid_dim),
            )
        self.hidden_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            )
        self.graph_conv1 = nn.Sequential(
            SAGEConv(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            )
        self.graph_conv2 = nn.Sequential(
            SAGEConv(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        feat = F.relu(self.hidden_linear(feat))
        out_feat = self.graph_conv1(feat, edge_index)
        out_feat = self.graph_conv2(out_feat, edge_index)
        return feat, out_feat

r"""
I replace the VanillaStellarNormedLinear from the paper with the LayerNorm, which is not strictly equivalent in computations
"""
class CustomStellarClassifficationHead_0(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    Replaces the VanillaStellarNormedLinear with LayerNorm
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 temperature: float
                 ):
        super(CustomStellarClassifficationHead_0, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.LayerNorm(num_classes)
        )
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out * self.temperature


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