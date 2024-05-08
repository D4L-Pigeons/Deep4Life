from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader, RandomNodeLoader, NeighborLoader
from torch_geometric.nn import (
    APPNP,
    AGNNConv,
    AntiSymmetricConv,
    ARMAConv,
    CGConv,
    ChebConv,
    ClusterGCNConv,
    CuGraphGATConv,
    CuGraphRGCNConv,
    CuGraphSAGEConv,
    DirGNNConv,
    DNAConv,
    DynamicEdgeConv,
    EdgeConv,
    EGConv,
    FAConv,
    FastRGCNConv,
    FeaStConv,
    FiLMConv,
    FusedGATConv,
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCN2Conv,
    GCNConv,
    GENConv,
    GeneralConv,
    GINConv,
    GINEConv,
    GMMConv,
    GPSConv,
    GraphConv,
    GravNetConv,
    HANConv,
    HEATConv,
    HeteroConv,
    HGTConv,
    HypergraphConv,
    LEConv,
    LGConv,
    MFConv,
    MixHopConv,
    NNConv,
    PANConv,
    PDNConv,
    PNAConv,
    PointGNNConv,
    PointNetConv,
    PointTransformerConv,
    PPFConv,
    ResGatedGraphConv,
    RGATConv,
    RGCNConv,
    SAGEConv,
    SGConv,
    SignedConv,
    SplineConv,
    SSGConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
    WLConv,
    WLConvContinuous,
    XConv,
)
from tqdm import tqdm

from datasets.stellar_data import StellarDataloader, make_graph_list_from_anndata
from models.ModelBase import ModelBase
from models.vanilla_stellar import VanillaStellarClassifficationHead
from utils import calculate_batch_accuracy


class CustomStellarEncoder(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.

    Adds the BatchNorm after each layer of the VanillaStellarEncoder
    """

    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        graph_conv_constructor: nn.Module,
        n_hidden_layers: int = 1,
        n_graph_layers: int = 1,
        batch_norm: bool = False,
    ):
        super(CustomStellarEncoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Linear(input_dim, hid_dim)
        self.hidden_linear = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU())
                for _ in range(n_hidden_layers)
            ]
        )
        self.graph_convs = nn.ModuleList()
        self.batch_norm = batch_norm
        self.batch_norms = nn.ModuleList()
        for _ in range(n_graph_layers):
            self.graph_convs.append(graph_conv_constructor(hid_dim, hid_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        feat = self.hidden_linear(feat)
        out_feat = feat
        for i, layer in enumerate(self.graph_convs):
            out_feat = layer(out_feat, edge_index)
            if self.batch_norm:
                out_feat = self.batch_norms[i](out_feat)
        return feat, out_feat


GRAPH_CONV_IMPLEMENTATIONS = {
    "GCNConv": GCNConv,  # 0
    "SAGEConv": SAGEConv,  # 2
    "GraphConv": GraphConv,  # 3
    "GatedGraphConv": GatedGraphConv,  # 5
    "ResGatedGraphConv": ResGatedGraphConv,  # 6
    "GATConv": GATConv,  # 7
    "GATv2Conv": GATv2Conv,  # 9
    "TransformerConv": TransformerConv,  # 10
    "AGNNConv": AGNNConv,  # 11
    "TAGConv": TAGConv,  # 12
    "ARMAConv": ARMAConv,  # 15
    "SGConv": SGConv,  # 16
    "MFConv": MFConv,  # 19
    "FeaStConv": FeaStConv,  # 34
    "HypergraphConv": HypergraphConv,  # 36
    "LEConv": LEConv,  # 37
    "ClusterGCNConv": ClusterGCNConv,  # 39
    "GENConv": GENConv,  # 40
    "FiLMConv": FiLMConv,  # 45
    "SuperGATConv": SuperGATConv,  # 46
}


r"""
I replace the VanillaStellarNormedLinear from the paper with the LayerNorm, which is not strictly equivalent in computations
"""


class CustomStellarClassifficationHead(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    Replaces the VanillaStellarNormedLinear with LayerNorm
    """

    def __init__(self, input_dim: int, num_classes: int, temperature: float):
        super(CustomStellarClassifficationHead, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out * self.temperature


class CustomSimpleStellarClassifficationHead(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    It is just linear layer.
    """

    def __init__(self, input_dim: int, num_classes: int, temperature: float):
        super(CustomSimpleStellarClassifficationHead, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out


CLASSIFICATION_HEAD_IMPLEMENTATIONS = [
    VanillaStellarClassifficationHead,
    CustomStellarClassifficationHead,
    CustomSimpleStellarClassifficationHead,
]


class CustomStellarModel(nn.Module):
    r"""
    A model that uses a graph encoder followed by a classification head to make predictions.

    Allows for flexibly altering the VanillaStellarReduced
    """

    def __init__(
        self,
        graph_conv_constructor: nn.Module,
        n_hidden_layers: int,
        n_graph_layers: int,
        batch_norm: bool,
        fc_net_constructor: nn.Module,
        input_dim: int,
        hid_dim: int,
        num_classes: int,
        temperature: float = 10.0,
    ):
        super(CustomStellarModel, self).__init__()
        self.encoder = CustomStellarEncoder(
            input_dim,
            hid_dim,
            graph_conv_constructor,
            n_hidden_layers,
            n_graph_layers,
            batch_norm,
        )
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
            GRAPH_CONV_IMPLEMENTATIONS[cfg.graph_conv_impl],
            cfg.n_hidden_layers,
            cfg.n_graph_layers,
            cfg.batch_norm,
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[cfg.classification_head_impl],
            cfg.input_dim,
            cfg.hid_dim,
            cfg.num_classes,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def train(self, data: anndata.AnnData) -> None:
        self.model = CustomStellarModel(
            GRAPH_CONV_IMPLEMENTATIONS[self.cfg.graph_conv_impl],
            self.cfg.n_hidden_layers,
            self.cfg.n_graph_layers,
            self.cfg.batch_norm,
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[self.cfg.classification_head_impl],
            self.cfg.input_dim,
            self.cfg.hid_dim,
            self.cfg.num_classes,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        graphs = make_graph_list_from_anndata(data, self.cfg.distance_threshold)
        batched_graphs = Batch.from_data_list(graphs)

        if self.cfg.batch_type == "graph":
            train_data_loader = StellarDataloader(
                graphs, batch_size=self.cfg.batch_size
            )
            self._train_graph_batch(train_data_loader, self.cfg.epochs)
        elif self.cfg.batch_type == "neighbors":
            train_data_loader = NeighborLoader(
                batched_graphs,
                num_neighbors=[5],
                batch_size=self.cfg.node_batch_size,
                shuffle=True,
            )
            self._train_graph_batch(train_data_loader, self.cfg.epochs)
        elif self.cfg.batch_type == "nodes_in_graph":
            train_data_loader = StellarDataloader(
                graphs, batch_size=self.cfg.batch_size
            )
            self._train_node_batch(train_data_loader, self.cfg.epochs)
        elif self.cfg.batch_type == "random_nodes":
            train_data_loader = RandomNodeLoader(
                batched_graphs,
                num_parts=batched_graphs.x.shape[0] // self.cfg.node_batch_size + 1,
                shuffle=True,
            )
            self._train_graph_batch(train_data_loader, self.cfg.epochs)

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        graphs = make_graph_list_from_anndata(data, self.cfg.distance_threshold)
        batched_graphs = Batch.from_data_list(graphs)
        cell_ids = np.concatenate(batched_graphs.cell_ids)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batched_graphs)
        preds = logits.argmax(dim=1).detach().numpy()
        pred_labels = data.obs["cell_labels"].cat.categories[preds]

        return (
            pd.Series(data=pred_labels, index=cell_ids)
            .reindex(data.obs.index)
            .to_numpy()
        )

    def predict_proba(self, data: anndata.AnnData) -> np.ndarray:
        graphs = make_graph_list_from_anndata(data, self.cfg.distance_threshold)
        batched_graphs = Batch.from_data_list(graphs)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batched_graphs)
        probs = F.softmax(logits, dim=1).detach().numpy()
        return probs

    def save(self, file_path: str) -> str:
        save_path = file_path + ".pth"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, file_path: str) -> None:
        self.model.load_state_dict(torch.load(file_path + ".pth"))

    def _train_graph_batch(
        self,
        train_loader: DataLoader,
        epochs: int,
        valid_loader: Union[DataLoader, None] = None,
        return_valid_acc: bool = False,
    ) -> Optional[float]:
        r"""
        Trains the model in a supervised manner.
        """
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            sum_loss = 0.0
            acc_sum = 0.0
            self.model.train()
            train_progress_bar = tqdm(
                train_loader, desc=f"Training - epoch {epoch}", leave=True
            )
            for batch_number, batch in enumerate(train_progress_bar):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(batch)
                loss = cross_entropy_loss_fn(output, batch.y)
                sum_loss += loss.item()
                acc_sum += calculate_batch_accuracy(output, batch.y)
                loss.backward()
                self.optimizer.step()
                train_progress_bar.set_postfix(
                    {
                        "Loss": sum_loss / (batch_number + 1),
                        "Accuracy": acc_sum / (batch_number + 1),
                    }
                )

            if valid_loader != None:
                val_loss_sum = 0
                val_acc_sum = 0
                self.model.eval()
                val_progress_bar = tqdm(
                    valid_loader, desc=f"Validation - epoch {epoch}", leave=True
                )
                for batch_number, batch in enumerate(val_progress_bar):
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        output, _ = self.model(batch)
                    loss = F.cross_entropy(output, batch.y)
                    val_loss_sum += loss.item()
                    val_acc_sum += calculate_batch_accuracy(output, batch.y)
                    val_progress_bar.set_postfix(
                        {
                            "Loss": val_loss_sum / (batch_number + 1),
                            "Accuracy": val_acc_sum / (batch_number + 1),
                        }
                    )

        if return_valid_acc:
            return val_acc_sum / (batch_number + 1)

    def _train_node_batch(
        self,
        train_loader: DataLoader,
        epochs: int,
        valid_loader: Union[DataLoader, None] = None,
        return_valid_acc: bool = False,
    ) -> Optional[float]:
        r"""
        Trains the model in a supervised manner.
        """
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            sum_loss = 0.0
            acc_sum = 0.0
            n = 0
            self.model.train()
            train_progress_bar = tqdm(
                train_loader, desc=f"Training - epoch {epoch}", leave=True
            )
            for batch_number, batch in enumerate(train_progress_bar):
                num_parts = len(batch.y) // self.cfg.node_batch_size + 1
                rnl = RandomNodeLoader(batch, num_parts=num_parts, shuffle=True)
                for part in rnl:
                    n += 1
                    part = part.to(self.device)
                    self.optimizer.zero_grad()
                    output, _ = self.model(part)
                    loss = cross_entropy_loss_fn(output, part.y)
                    sum_loss += loss.item()
                    acc_sum += calculate_batch_accuracy(output, part.y)
                    loss.backward()
                    self.optimizer.step()
                train_progress_bar.set_postfix(
                    {"Loss": sum_loss / n, "Accuracy": acc_sum / n}
                )

            if valid_loader != None:
                val_loss_sum = 0
                val_acc_sum = 0
                self.model.eval()
                val_progress_bar = tqdm(
                    valid_loader, desc=f"Validation - epoch {epoch}", leave=True
                )
                for batch_number, batch in enumerate(val_progress_bar):
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        output, _ = self.model(batch)
                    loss = F.cross_entropy(output, batch.y)
                    val_loss_sum += loss.item()
                    val_acc_sum += calculate_batch_accuracy(output, batch.y)
                    val_progress_bar.set_postfix(
                        {
                            "Loss": val_loss_sum / (batch_number + 1),
                            "Accuracy": val_acc_sum / (batch_number + 1),
                        }
                    )

        if return_valid_acc:
            return val_acc_sum / (batch_number + 1)
