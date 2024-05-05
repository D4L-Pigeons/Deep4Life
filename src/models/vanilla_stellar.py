import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch import Tensor
from anndata import AnnData
from typing import Optional, Tuple, Generator
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from tqdm import tqdm
from models.ModelBase import ModelBase
from datasets.stellar_data import StellarDataloader, make_graph_list_from_anndata
from utils import calculate_entropy_logits, calculate_entropy_probs, calculate_batch_accuracy, MarginLoss
from itertools import cycle
import anndata
import pandas as pd

class VanillaStellarNormedLinear(nn.Module):
    r"""
    A linear layer that normalizes the input and the weight before the matrix multiplication.
    """
    def __init__(self, in_features: int, out_features: int, temperature: float):
        super(VanillaStellarNormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        assert type(x) == Tensor, f"Expected Tensor, got {type(x)}"
        # print(type(x), type(self.weight))

        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.temperature * out


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
    def __init__(self, input_dim: int, num_classes: int, temperature: float):
        super(VanillaStellarClassifficationHead, self).__init__()
        self.linear = VanillaStellarNormedLinear(input_dim, num_classes, temperature=temperature)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out


class VanillaStellarFCNet(nn.Module):
    r"""
    A fully connected network that uses a linear layer to make predictions.
    """
    def __init__(self, input_dim: int, num_classes: int, temperature: float=10.0):
        super(VanillaStellarFCNet, self).__init__()
        self.classifier = VanillaStellarClassifficationHead(input_dim, num_classes, temperature=temperature)

    def forward(self, data: Data):
        x = data.x
        out = self.classifier(x)
        return out


class VanillaStellarModel(nn.Module):
    r"""
    A model that uses a graph encoder followed by a classification head to make predictions.
    """
    def __init__(self, input_dim: int, hid_dim: int, num_classes: int, temperature: float=10.0):
        super(VanillaStellarModel, self).__init__()
        self.encoder = VanillaStellarEncoder(input_dim, hid_dim)
        self.fc_net = VanillaStellarClassifficationHead(hid_dim, num_classes, temperature=temperature)

    def forward(self, data: Data):
        _, out_feat = self.encoder(data)
        assert type(out_feat) == Tensor, f"Expected Tensor, got {type(out_feat)}"
        out = self.fc_net(out_feat)
        return out, out_feat


class VanillaStellar:
    r"""
    A class that encapsulates the model, optimizer and training loop.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = VanillaStellarModel(cfg.input_dim, cfg.hid_dim, cfg.num_standard_classes + cfg.num_seed_classes).to(self.device) # The model has the standard classes and the seed classes
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.seed_model = VanillaStellarFCNet(cfg.input_dim, cfg.num_standard_classes).to(self.device)
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
            train_progress_bar = tqdm(train_loader, desc=f"Training - epoch {epoch}", leave=False)
            for batch in train_progress_bar:
                batch = batch.to(self.device)
                output = self.seed_model(batch)
                loss = cross_entropy_loss_fn(output, batch.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item()
                train_acc_sum += calculate_batch_accuracy(output, batch.y)
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
            epochs: int,
            ) -> None:
        r"""
        Trains the model using the Stellar algorithm.
        """
        # binary_cross_entropy_loss_fn = nn.BCEWithLogitsLoss()
        sum_loss = 0.0
        for epoch in range(epochs):
            n: int = 0
            mean_uncert = self.calculate_prediction_uncertainty_on_unlabeled(extended_unlabeled_loader)
            margin_loss_fn = MarginLoss(m=-mean_uncert)
            self.model.train()
            # As this is vanilla Stellar I follow the original implementation with strange way of iterating over the unlabeled data. The assumptions is that the unlabeled dataloader has fewer batches.
            # cycle is not doing the shuffling
            train_progress_bar = tqdm(zip(labeled_loader, cycle(extended_unlabeled_loader)), desc=f"Training - epoch {epoch}", leave=False)
            for lab_graph_batch, ulab_graph_batch in train_progress_bar:
                n += len(lab_graph_batch.x)
                # ulab_graph_batch, ulab_novel_label_seed_idx = ulab_batch
                ulab_ce_idx = torch.where(ulab_graph_batch.novel_label_seeds > 0)[0]
                # lab_graph_batch.x, ulab_graph_batch.x = lab_graph_batch.x.to(self.device), ulab_graph_batch.x.to(self.device)
                lab_graph_batch, ulab_graph_batch = lab_graph_batch.to(self.device), ulab_graph_batch.to(self.device)
                self.optimizer.zero_grad()
                
                lab_output, lab_feat = self.model(lab_graph_batch)
                ulab_output, ulab_feat = self.model(ulab_graph_batch)
                output = torch.cat([lab_output, ulab_output], dim=0)
                prob = F.softmax(output, dim=1)
                entropy_loss = calculate_entropy_probs(prob.mean(dim=0))
                
                lab_len = len(lab_graph_batch)
                ulab_len = len(ulab_graph_batch)
                batch_size = lab_len + ulab_len
                
                feat = torch.cat([lab_feat, ulab_feat], dim=0)

                # print(f"The shape of the feature is {feat.shape}")
                feat_norm = F.normalize(feat)
                cos_dist = torch.matmul(feat_norm, feat_norm.T)#F.cosine_similarity(feat, feat, dim=1)
                # print(f"The shape of the cosine distance is {cos_dist.shape}")

                target = lab_graph_batch.y

                pos_pairs = []
                for i in range(lab_len):
                    idxs = torch.where(target == target[i])[0].tolist()
                    idxs.remove(i)
                    if len(idxs) == 0:
                        pos_pairs.append(i)
                    else:
                        pos_pairs.append(idxs[torch.randint(0, len(idxs), (1,)).item()])
                
                ulab_cos_dist = cos_dist[lab_len:, :]
                pos_idx = torch.topk(ulab_cos_dist, k=2, dim=1)[1].flatten().tolist()
                pos_pairs.extend(pos_idx)

                pos_prob = prob[pos_pairs, :]
                pos_sim = (prob**2).mean(dim=1)
                # binary_cross_entropy_loss_fn(pos_sim, torch.ones_like(pos_sim))
                bce_loss = -pos_sim.log().mean().item()
                ce_idx = torch.cat([torch.arange(lab_len), lab_len + ulab_ce_idx], dim=0)
                target_ext = torch.cat([target, ulab_graph_batch.novel_label_seeds])
                # print(f"The len of the output is {len(output)} and the len of the target is {len(target_ext)}")
                margin_loss = margin_loss_fn(output[ce_idx], target_ext[ce_idx])
                # print(f"The type of margin loss is {type(margin_loss.item())}")
                # print(f"The type of entropy loss is {type(entropy_loss.item())}")
                # print(f"The type of bce loss is {type(bce_loss)}")
                print(f"bce_loss: {bce_loss}, margin_loss: {margin_loss.item()}, entropy_loss: {entropy_loss.item()}")
                loss = self.cfg.bce_coeff * bce_loss + self.cfg.margin_coeff * margin_loss - self.cfg.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                # print(f"The type of loss is {type(loss.item())}")
                sum_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                # print(f"The len of the train progress bar is {n}")
                train_progress_bar.set_postfix({"Loss": sum_loss / n})
    
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
        batch_list = []
        largest_standard_class_label = self.cfg.num_standard_classes - 1
        self.seed_model.eval()
        for batch in unlabeled_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.seed_model(batch)
            entr = calculate_entropy_logits(output, dim=1)
            
            clusters, max_cluster_label = self.find_clusters(batch)
            clusters_entropy = np.zeros(max_cluster_label + 1)
            for cluster_label in range(max_cluster_label + 1):
                cluster_indices = (clusters == cluster_label)
                clusters_entropy[cluster_label] = entr[cluster_indices].mean()
            
            novel_cluster_idxs = np.argsort(clusters_entropy)[-num_seed_class:]
            novel_label_seeds = torch.zeros_like(clusters)
            for i, novel_cluster_idx in enumerate(novel_cluster_idxs):
                novel_label_seeds[clusters == novel_cluster_idx] = largest_standard_class_label + i + 1
            
            batch.novel_label_seeds = novel_label_seeds
            # yield batch, novel_label_seeds
            batch_list.append(batch)

        return DataLoader(batch_list, batch_size=unlabeled_loader.batch_size, shuffle=True)
           
    def predict(self, data: Data) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            data.x = data.x.to(self.device)
            output, _ = self.model(data)
        probs = F.softmax(output, dim=1)
        # print(probs)
        confs, preds = probs.max(dim=1)
        return confs, preds
    
    def calculate_prediction_uncertainty_on_unlabeled(self, dataloader: DataLoader) -> Tensor:
        confs_sum: float = 0.0
        n: int = 0
        for batch in dataloader:
            # graph_batch, _ = batch
            batch = batch.to(self.device)
            n += len(batch.x)
            # print(type(batch), type(graph_batch))
            # graph_batch.x = graph_batch.x.to(self.device)
            confs, preds = self.predict(batch)
            confs_sum += confs.sum().item()
        # print(confs_sum, n)
        mean_uncertainty = 1 - confs_sum / n
        return mean_uncertainty

    def train_stellar(
            self,
            labeled_loader: DataLoader,
            unlabeled_loader: DataLoader,
            epochs: int,
            seed_epochs: int
            ): #-> None:
        r"""
        Trains the model using the Stellar algorithm.
        """
        self.train_seed_model(labeled_loader, seed_epochs)
        novel_label_seeds_unlabeled_loader = self.estimate_seeds(unlabeled_loader, self.cfg.num_seed_classes)
        self.train_main_model(labeled_loader, novel_label_seeds_unlabeled_loader, epochs)
        # return novel_label_seeds_unlabeled_loader


class VanillaStellarReduced(ModelBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = VanillaStellarModel(cfg.input_dim, cfg.hid_dim, cfg.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.label_encoder = LabelEncoder().fit(cfg.target_labels)

    def train(self, data: anndata.AnnData) -> None:
        self.model = VanillaStellarModel(self.cfg.input_dim, self.cfg.hid_dim, self.cfg.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        
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