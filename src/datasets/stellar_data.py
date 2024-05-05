import numpy as np
import torch
from datasets import load_d4ls
from pathlib import Path
from typing import List
from torch_geometric.data import Data
import torch_geometric.loader
from sklearn.preprocessing import LabelEncoder
import anndata


def _get_distances(pos: np.ndarray) -> np.ndarray:
    r"""
    Get the distances between the provided points
    """
    return np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)


def get_all_distances():
    r"""
    Get all distances between cells in the dataset. This is useful for setting the distance threshold for the graph construction.
    """
    anndata = load_d4ls.load_full_anndata()
    all_distances = np.array([])

    # GROUPING BY INDICATION MAY BE A GOOD IDEA FOR TESTING TRANSFER LEARNING
    for sample_id in anndata.obs["sample_id"].unique():
        sample_cell_indices = (anndata.obs["sample_id"] == sample_id).values
        sample_pos = anndata.obs[sample_cell_indices][["Pos_X", "Pos_Y"]].values.astype(np.float32)
        distances = _get_distances(sample_pos)
        all_distances = np.concatenate([all_distances, distances.flatten()])
    
    return all_distances  


def _get_edges(
        pos: np.ndarray,
        distance_threshold: float
        ) -> np.ndarray:
    r"""
    Get the edges between the cells based on the distance threshold
    """
    distances = _get_distances(pos)
    neighbourhood_mask = distances <= distance_threshold
    np.fill_diagonal(neighbourhood_mask, 0)
    edges = np.array(np.where(neighbourhood_mask))
    return edges
    
def make_graph_list_from_anndata(
        anndata: anndata.AnnData,
        distance_threshold: float,
        obs_feature_names: list[str] = []
        ) -> list:
    r"""
    Make a list of graphs from the dataset.
    """
    targets = anndata.obs["cell_labels"].cat.codes.to_numpy()
    
    graphs = []
    # GROUPING BY INDICATION MAY BE A GOOD IDEA FOR TESTING TRANSFER LEARNING
    for sample_id in anndata.obs["sample_id"].unique():
        sample_cell_indices = (anndata.obs["sample_id"] == sample_id).values
        sample_cell_ids = np.array(anndata.obs[sample_cell_indices].index)
        
        sample_pos = anndata.obs[sample_cell_indices][["Pos_X", "Pos_Y"]].values.astype(np.float32)
        sample_edges = _get_edges(sample_pos, distance_threshold)

        sample_targets = targets[sample_cell_indices].astype(np.int32)
        
        sample_obs_features = anndata.obs[sample_cell_indices][obs_feature_names].values.astype(np.float32)
        sample_expressions = anndata.layers['exprs'][sample_cell_indices].astype(np.float32)
        features = np.concatenate([sample_obs_features, sample_expressions], axis=-1)
        del sample_obs_features, sample_expressions
        
        sample_graph = Data(
            x=torch.FloatTensor(features),
            edge_index=torch.LongTensor(sample_edges),
            y=torch.LongTensor(sample_targets),
            cell_ids=sample_cell_ids
            )
        
        graphs.append(sample_graph)
    
    return graphs


def make_graph_list(
        obs_feature_names: List[str],
        save_filename: str,
        distance_threshold: float,
        test: bool = False
        ) -> list:
    r"""
    Make a list of graphs from the dataset and save it to a file.
    """
    data_path = load_d4ls.TRAIN_DATA_PATH if not test else load_d4ls.TEST_DATA_PATH
    file_path = data_path / save_filename
    assert file_path.exists() == False, f"File {save_filename} already exists"
    
    anndata = load_d4ls.load_full_anndata()
    le = LabelEncoder()
    targets = le.fit_transform(anndata.obs["cell_labels"].values)
    graphs = []
    # GROUPING BY INDICATION MAY BE A GOOD IDEA FOR TESTING TRANSFER LEARNING
    for sample_id in anndata.obs["sample_id"].unique():
        sample_cell_indices = (anndata.obs["sample_id"] == sample_id).values
        sample_cell_ids = np.array(anndata.obs[sample_cell_indices].index)
        
        sample_pos = anndata.obs[sample_cell_indices][["Pos_X", "Pos_Y"]].values.astype(np.float32)
        sample_edges = _get_edges(sample_pos, distance_threshold)

        sample_targets = targets[sample_cell_indices].astype(np.int32)
        
        sample_obs_features = anndata.obs[sample_cell_indices][obs_feature_names].values.astype(np.float32)
        sample_expressions = anndata.layers['exprs'][sample_cell_indices].astype(np.float32)
        features = np.concatenate([sample_obs_features, sample_expressions], axis=-1)
        del sample_obs_features, sample_expressions
        
        sample_graph = Data(
            x=torch.FloatTensor(features),
            edge_index=torch.LongTensor(sample_edges),
            y=torch.LongTensor(sample_targets),
            cell_ids=sample_cell_ids
            )
        graphs.append(sample_graph)
    
    torch.save(obj=graphs, f=file_path)


class StellarDataloader(torch_geometric.loader.DataLoader):
    r"""
    DataLoader for the StellarGraph dataset
    """
    def __init__(self, graphs: list[Data], batch_size: int, shuffle: bool = True, graphs_idx: List[int] | None = None):
        graphs = [graph for idx, graph in enumerate(graphs) if graphs_idx == None or idx in graphs_idx]
        super().__init__(graphs, batch_size=batch_size, shuffle=shuffle)
       
    @classmethod 
    def from_file(cls, filename, test: bool, batch_size: int, shuffle: bool = True, graphs_idx: List[int] | None = None):
        data_path = load_d4ls.TEST_DATA_PATH if test else load_d4ls.TRAIN_DATA_PATH
        file_path = data_path / filename
        assert file_path.exists(), f"File {filename} does not exist in {data_path}"
        graphs = torch.load(file_path)
        return cls(graphs, batch_size, shuffle, graphs_idx)
