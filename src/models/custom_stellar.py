from models.vanilla_stellar import VanillaStellarEncoder, VanillaStellarClassifficationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, CuGraphSAGEConv, GraphConv, GravNetConv, GatedGraphConv, ResGatedGraphConv, GATConv, CuGraphGATConv, FusedGATConv, GATv2Conv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, SSGConv, APPNP, MFConv, RGCNConv, FastRGCNConv, CuGraphRGCNConv, RGATConv, SignedConv, DNAConv, PointNetConv, GMMConv, SplineConv, NNConv, CGConv, EdgeConv, DynamicEdgeConv, XConv, PPFConv, FeaStConv, PointTransformerConv, HypergraphConv, LEConv, PNAConv, ClusterGCNConv, GENConv, GCN2Conv, PANConv, WLConv, WLConvContinuous, FiLMConv, SuperGATConv, FAConv, EGConv, PDNConv, GeneralConv, HGTConv, HEATConv, HeteroConv, HANConv, LGConv, PointGNNConv, GPSConv, AntiSymmetricConv, DirGNNConv, MixHopConv
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader, RandomNodeLoader
from typing import Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from models.ModelBase import ModelBase
from datasets.stellar_data import StellarDataloader, make_graph_list_from_anndata
from utils import calculate_batch_accuracy
import anndata
import pandas as pd
from typing import Union


class CustomStellarEncoder(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.

    Adds the BatchNorm after each layer of the VanillaStellarEncoder
    """
    def __init__(self,
                 input_dim: int,
                 hid_dim: int,
                 graph_conv_constructor: nn.Module,
                 n_graph_layers: int=2,
                 batch_norm: bool=False,
                 ):
        super(CustomStellarEncoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Linear(input_dim, hid_dim)
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
        out_feat = feat
        for i, layer in enumerate(self.graph_convs):
            out_feat = layer(out_feat, edge_index)
            if self.batch_norm:
                out_feat = self.batch_norms[i](out_feat)
        return feat, out_feat


GRAPH_CONV_IMPLEMENTATIONS = [
    GCNConv, # 0
    ChebConv, # 1 ChebConv.__init__() missing 1 required positional argument: 'K'
    SAGEConv, # 2
    GraphConv, # 3
    GravNetConv, # 4  #GravNetConv.__init__() missing 3 required positional arguments: 'space_dimensions', 'propagate_dimensions', and 'k'
    GatedGraphConv, # 5
    ResGatedGraphConv, # 6
    GATConv, # 7
    FusedGATConv, # 8
    GATv2Conv,  # 9
    TransformerConv, # 10
    AGNNConv, # 11
    TAGConv, # 12
    GINConv, # 13
    GINEConv, # 14
    ARMAConv, # 15
    SGConv, # 16
    SSGConv, # 17
    APPNP, # 18
    MFConv, # 19
    RGCNConv, # 20
    FastRGCNConv, # 21
    RGATConv, # 22
    SignedConv, # 23
    DNAConv, # 24
    PointNetConv, # 25
    GMMConv, # 26
    SplineConv, # 27
    NNConv, # 28
    CGConv, # 29
    EdgeConv, # 30
    DynamicEdgeConv, # 31
    XConv, # 32
    PPFConv, # 33
    FeaStConv, # 34
    PointTransformerConv, # 35
    HypergraphConv, # 36
    LEConv, # 37
    PNAConv, # 38
    ClusterGCNConv, # 39
    GENConv, # 40
    GCN2Conv,   # 41
    PANConv,   # 42
    WLConv,  # 43
    WLConvContinuous, # 44
    # FiLMConv, # 45
    lambda in_channels, out_channels: FiLMConv(in_channels, out_channels, 1), # 45
    SuperGATConv, # 46
    FAConv, # 47
    EGConv, # 48
    PDNConv, # 49
    GeneralConv, # 50
    HGTConv, # 51
    HEATConv, # 52
    HeteroConv, # 53
    HANConv, # 54
    LGConv, # 55
    PointGNNConv, # 56
    GPSConv, # 57
    AntiSymmetricConv, # 58
    DirGNNConv, # 59
    MixHopConv # 60
]


r"""
I replace the VanillaStellarNormedLinear from the paper with the LayerNorm, which is not strictly equivalent in computations
"""
class CustomStellarClassifficationHead(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.

    Replaces the VanillaStellarNormedLinear with LayerNorm
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 temperature: float
                 ):
        super(CustomStellarClassifficationHead, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            # nn.LayerNorm(num_classes)
        )
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out * self.temperature


CLASSIFICATION_HEAD_IMPLEMENTATIONS = [
    VanillaStellarClassifficationHead,
    CustomStellarClassifficationHead
]


class CustomStellarModel(nn.Module):
    r"""
    A model that uses a graph encoder followed by a classification head to make predictions.

    Allows for flexibly altering the VanillaStellarReduced
    """
    def __init__(self,
                 graph_conv_constructor: nn.Module,
                 n_graph_layers: int,
                 batch_norm: bool,
                 fc_net_constructor: nn.Module,
                 input_dim: int,
                 hid_dim: int,
                 num_classes: int,
                 temperature: float=10.0
                 ):
        super(CustomStellarModel, self).__init__()
        self.encoder = CustomStellarEncoder(input_dim, hid_dim, graph_conv_constructor, n_graph_layers, batch_norm)
        self.fc_net = fc_net_constructor(hid_dim, num_classes, temperature=temperature)

    def forward(self, data: Data):
        _, out_feat = self.encoder(data)
        assert type(out_feat) == Tensor, f"Expected Tensor, got {type(out_feat)}"
        out = self.fc_net(out_feat)
        return out, out_feat


class CustomStellarModel2(nn.Module):
    def __init__(self, input_dim, hid_dim, num_classes):
        super(CustomStellarModel2, self).__init__()
        self.linear = nn.Linear(input_dim, hid_dim)
        # self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.graph_conv1 = FiLMConv(hid_dim, hid_dim)
        self.inst_norm = pyg_nn.norm.InstanceNorm(hid_dim)
        self.graph_conv2 = ResGatedGraphConv(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.inst_norm(x)
        x = F.relu(self.linear(x))
        # x = F.relu(self.linear2(x))
        x = F.relu(self.graph_conv1(x, edge_index))
        x = F.relu(self.graph_conv2(x, edge_index))
        # x = self.inst_norm(x)
        # x = F.relu(x)
        # x = self.asapool(x, data.batch)
        x = self.fc(x)
        return x, x

class CustomStellarReduced(ModelBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = CustomStellarModel(
            GRAPH_CONV_IMPLEMENTATIONS[cfg.graph_conv_impl],
            cfg.n_graph_layers,
            cfg.batch_norm,
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[cfg.classification_head_impl],
            cfg.input_dim,
            cfg.hid_dim,
            cfg.num_classes
            ).to(self.device)
        # self.model = CustomStellarModel2(cfg.input_dim, cfg.hid_dim, cfg.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self, data: anndata.AnnData) -> None:
        self.model = CustomStellarModel(
            GRAPH_CONV_IMPLEMENTATIONS[self.cfg.graph_conv_impl],
            self.cfg.n_graph_layers,
            self.cfg.batch_norm,
            CLASSIFICATION_HEAD_IMPLEMENTATIONS[self.cfg.classification_head_impl],
            self.cfg.input_dim,
            self.cfg.hid_dim,
            self.cfg.num_classes
            ).to(self.device)
        # self.model = CustomStellarModel2(self.cfg.input_dim, self.cfg.hid_dim, self.cfg.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        graphs = make_graph_list_from_anndata(data, self.cfg.distance_threshold)
        train_data_loader = StellarDataloader(graphs, batch_size=self.cfg.batch_size)
        
        self._train(train_data_loader, self.cfg.epochs)

    def predict(self, data: anndata.AnnData) -> np.ndarray:
        graphs = make_graph_list_from_anndata(data, self.cfg.distance_threshold)
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
            valid_loader: Union[DataLoader, None] = None,
            return_valid_acc: bool = False
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
            train_progress_bar = tqdm(train_loader, desc=f"Training - epoch {epoch}", leave=True)
            for batch_number, batch in enumerate(train_progress_bar):
                # batch = batch.to(self.device)
                # self.optimizer.zero_grad()
                # output, _ = self.model(batch)
                # loss = cross_entropy_loss_fn(output, batch.y)
                # sum_loss += loss.item()
                # acc_sum += calculate_batch_accuracy(output, batch.y)
                # loss.backward()
                # self.optimizer.step()
                # train_progress_bar.set_postfix({"Loss": sum_loss / (batch_number + 1), "Accuracy": acc_sum / (batch_number + 1)})
                BATCH_SIZE = 200
                num_parts = len(batch.y) // BATCH_SIZE
                rnl = RandomNodeLoader(batch, num_parts=num_parts)
                for part in rnl:
                    n += 1
                    part = part.to(self.device)
                    self.optimizer.zero_grad()
                    output, _ = self.model(part)
                    # print(f"output: {output.shape}, y: {part.y.shape}")
                    loss = cross_entropy_loss_fn(output, part.y)
                    sum_loss += loss.item()
                    acc_sum += calculate_batch_accuracy(output, part.y)
                    loss.backward()
                    self.optimizer.step()
                train_progress_bar.set_postfix({"Loss": sum_loss / n, "Accuracy": acc_sum / n})

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