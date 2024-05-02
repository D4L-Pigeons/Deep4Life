import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch import Tensor
from anndata import AnnData
from typing import Optional, Tuple, Generator
import numpy as np
import copy
import scanpy as sc
from tqdm import tqdm
from src.utils import calculate_entropy, calculate_batch_accuracy, MarginLoss
from itertools import cycle


class VanillaStellarNormedLinear(nn.Module):
    r"""
    A linear layer that normalizes the input and the weight before the matrix multiplication.
    """
    def __init__(self, in_features: int, out_features: int):
        super(VanillaStellarNormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x: Tensor) -> Tensor:
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out


class VanillaStellarEncoder(nn.Module):
    r"""
    A graph encoder that uses a GCN layer followed by a linear layer.
    """
    def __init__(self, input_dim: int, hid_dim: int=128):
        super(VanillaStellarEncoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_linear = nn.Linear(input_dim, hid_dim)
        self.graph_conv = SAGEConv(hid_dim, hid_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat = F.relu(self.input_linear(x))
        out_feat = self.graph_conv(feat, edge_index)
        return feat, out_feat


class VanillaStellarClassifficationHead(nn.Module):
    r"""
    A classification head that uses a linear layer to make predictions.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(VanillaStellarClassifficationHead, self).__init__()
        self.linear = VanillaStellarNormedLinear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out


class VanillaStellarFCNet(nn.Module):
    r"""
    A fully connected network that uses a linear layer to make predictions.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(VanillaStellarFCNet, self).__init__()
        self.classifier = VanillaStellarClassifficationHead(input_dim, num_classes)

    def forward(self, data: Data):
        x = data.x
        out = self.classifier(x)
        return out


class VanillaStellarModel(nn.Module):
    r"""
    A model that uses a graph encoder followed by a classification head to make predictions.
    """
    def __init__(self, input_dim: int, hid_dim: int, num_classes: int):
        super(VanillaStellarModel, self).__init__()
        self.encoder = VanillaStellarEncoder(input_dim, hid_dim)
        self.fc_net = VanillaStellarFCNet(hid_dim, num_classes)

    def forward(self, data: Data):
        _, out_feat = self.encoder(data)
        out = self.fc_net(out_feat)
        return out, out_feat


class VanillaStellar:
    r"""
    A class that encapsulates the model, optimizer and training loop.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = VanillaStellarModel(cfg.input_dim, cfg.hid_dim, cfg.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.seed_model = VanillaStellarFCNet(cfg.input_dim, cfg.num_classes).to(self.device)
        self.seed_optimizer = optim.Adam(self.seed_model.parameters(), lr=cfg.seed_lr)

    def train_seed_model(
            self,
            train_loader: DataLoader,
            epochs: int,
            val_loader: Optional[DataLoader] = None
            ) -> None:
        r"""
        Trains the model in a supervised manner.
        """
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        self.seed_model.train()
        for epoch in range(epochs):
            train_progress_bar = tqdm(train_loader, desc=f"Training - epoch {epoch}", leave=True)
            for batch in train_progress_bar:
                batch = batch.to(self.device)
                output = self.seed_model(batch)
                loss = cross_entropy_loss_fn(output, batch.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item()
                train_acc_sum += self.batch_accuracy(output, batch.y)
                train_progress_bar.set_postfix({"Loss": train_loss_sum / len(train_progress_bar), "Accuracy": train_acc_sum / len(train_progress_bar)})
            
            if val_loader is not None:
                val_loss_sum = 0
                val_acc_sum = 0
                self.seed_model.eval()
                val_progress_bar = tqdm(val_loader, desc=f"Validation - epoch {epoch}", leave=True)
                for batch in val_progress_bar:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        output = self.seed_model(batch)
                    loss = F.cross_entropy(output, batch.y)
                    val_loss_sum += loss.item()
                    val_acc_sum += calculate_batch_accuracy(output, batch.y)
                    val_progress_bar.set_postfix({"Loss": val_loss_sum / len(val_progress_bar), "Accuracy": val_acc_sum / len(val_progress_bar)})

    def train_main_model(
            self,
            labeled_loader: DataLoader,
            extended_unlabeled_loader: DataLoader,
            epochs: int
            ) -> None:
        r"""
        Trains the model using the Stellar algorithm.
        """
        binary_cross_entropy_loss_fn = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            self.model.train()
            # As this is vanilla Stellar I follow the original implementation with strange way of iterating over the unlabeled data. The assumptions is that the unlabeled dataloader has fewer batches.
            # cycle is not doing the shuffling
            train_progress_bar = tqdm(zip(labeled_loader, cycle(extended_unlabeled_loader)), desc=f"Training - epoch {epoch}", leave=True)
            for lab_graph_batch, ulab_batch in train_progress_bar:
                ulab_graph_batch, ulab_novel_label_seed_idx = ulab_batch
                lab_graph_batch.x, ulab_graph_batch.x = lab_graph_batch.x.to(self.device), ulab_graph_batch.x.to(self.device)
                self.optimizer.zero_grad()
                lab_output, lab_feat = self.model(lab_graph_batch)
                ulab_output, ulab_feat = self.model(ulab_graph_batch)
                lab_len = len(lab_graph_batch)
                ulab_len = len(ulab_graph_batch)
                batch_size = lab_len + ulab_len
                output = torch.cat([lab_output, ulab_output], dim=0)
                feat = torch.cat([lab_feat, ulab_feat], dim=0)

                prob = F.softmax(output, dim=1)

                cos_dist = F.cosine_similarity(feat, feat, dim=2)
                ulab_cos_dist = cos_dist[lab_len:, :]
                vals, pos_idx = torch.topk(ulab_cos_dist, k=2, dim=1)

                target = lab_graph_batch.y

                # for i in range(lab_len):


        
        # margin_loss_fn = MarginLoss(m=-)
    
    @staticmethod
    def find_clusters(data: Data) -> Tuple[Tensor, np.ndarray]:
        r"""
        Finds the clusters using the Louvain algorithm.
        """
        adata = AnnData(X=data.x.cpu().numpy())
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata)
        clusters = torch.tensor(adata.obs["louvain"].cat.codes.values)
        max_cluster_label = clusters.max().item()
        return clusters, max_cluster_label
    
    def estimate_seeds(
            self,
            unlabeled_loader: DataLoader,
            num_seed_class: int
            ) -> Generator[Tuple[Data, Tensor], None, None]:
        r"""
        Estimates the seeds using a fully connected network.
        """
        largest_standard_class_label = self.cfg.num_classes - 1
        self.seed_model.eval()
        for batch in unlabeled_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.seed_model(batch)
            entr = calculate_entropy(output)
            
            clusters, max_cluster_label = self.find_clusters(batch)
            clusters_entropy = np.zeros(max_cluster_label)
            for cluster_label in range(max_cluster_label + 1):
                cluster_indices = (clusters == cluster_label)
                clusters_entropy[cluster_label] = entr[cluster_indices].mean()
            
            novel_cluster_idxs = np.argsort(clusters_entropy)[-num_seed_class:]
            novel_label_seeds = torch.zeros_like(clusters)
            for i, novel_cluster_idx in enumerate(novel_cluster_idxs):
                novel_label_seeds[clusters == novel_cluster_idx] = largest_standard_class_label + i + 1
            
            yield batch, novel_label_seeds
           
    def predict(self, data: Data) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
        probs = F.softmax(output, dim=1)
        confs, preds = probs.max(dim=1)
        mean_uncertainty = 1 - confs.mean().item()
        return mean_uncertainty, preds
    
    def train_stellar(
            self,
            labeled_loader: DataLoader,
            unlabeled_loader: DataLoader,
            epochs: int,
            seed_epochs: int
            ) -> None:
        r"""
        Trains the model using the Stellar algorithm.
        """
        self.train_seed_model(labeled_loader, seed_epochs)
        novel_label_seeds_unlabeled_loader = self.estimate_seeds(unlabeled_loader, self.cfg.num_seed_class)
        