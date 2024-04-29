import numpy as np
import pandas as pd
import torch
from src.datasets import load_d4ls
from pathlib import Path
from typing import List

from torch_geometric.data import Data, Dataset

from sklearn.preprocessing import LabelEncoder

# class StellarDataset(Dataset):
#     def __init__(self, root: str, transform=None, pre_transform=None):
#         super(StellarDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return []

#     @property
#     def processed_file_names(self):
#         return []

#     def download(self):
#         pass

#     def process(self):
#         data_list = []
#         for data in self.data:
#             x = torch.tensor(data.x, dtype=torch.float)
#             edge_index = torch.tensor(data.edge_index, dtype=torch.long)
#             y = torch.tensor(data.y, dtype=torch.long)
#             cell_ids = data.cell_ids
#             data = Data(x=x, edge_index=edge_index, y=y, cell_ids=cell_ids)
#             data_list.append(data)
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def __len__(self):
#         return len(self.data)

#     def get(self, idx):
#         data = self.data[idx]
#         x = torch.tensor(data.x, dtype=torch.float)
#         edge_index = torch.tensor(data.edge_index, dtype=torch.long)
#         y = torch.tensor(data.y, dtype=torch.long)
#         cell_ids = data.cell_ids
#         return Data(x=x, edge_index=edge_index, y=y, cell_ids=cell_ids)

def create_stellar_dataset_from_cell_data(
        obs_feature_names: List[str],
        save_filename: str,
        distance_threshold: float,
        test: bool = False
        ) -> None:
    data_path = load_d4ls.TRAIN_DATA_PATH if not test else load_d4ls.TEST_DATA_PATH
    assert (data_path / save_filename).exists() == False, f"File {save_filename} already exists"
    
    anndata = load_d4ls.load_full_anndata()
    le = LabelEncoder()
    targets = le.fit_transform(anndata.obs["cell_labels"].values)
    graph_data = []
    # GROUPING BY INDICATION MAY BE A GOOD IDEA FOR TESTING TRANSFER LEARNING
    for sample_id in anndata.obs["sample_id"].unique():
        sample_cell_indices = (anndata.obs["sample_id"] == sample_id).values
        sample_cell_ids = anndata.obs[sample_cell_indices].index
        
        sample_pos = anndata.obs[sample_cell_indices][["Pos_X", "Pos_Y"]].values.astype(np.float32)
        neighbourhood_mask = np.linalg.norm(sample_pos[:, None] - sample_pos[None, :], axis=-1) <= distance_threshold
        sample_edges = np.array(np.where(neighbourhood_mask))#.T
        del sample_pos, neighbourhood_mask
        
        sample_obs_features = anndata.obs[sample_cell_indices][obs_feature_names].values.astype(np.float32)
        sample_targets = targets[sample_cell_indices].astype(np.int32)
        sample_expressions = anndata.layers['exprs'][sample_cell_indices].astype(np.float32)
        features = np.concatenate([sample_obs_features, sample_expressions], axis=-1)
        del sample_obs_features, sample_expressions
        
        graph = Data(x=torch.FloatTensor(features), edge_index=torch.LongTensor(sample_edges), y=torch.LongTensor(sample_targets), cell_ids=sample_cell_ids)
        graph_data.append(graph)
    
    return graph_data