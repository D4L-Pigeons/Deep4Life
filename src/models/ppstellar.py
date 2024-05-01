import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from utils import entropy, MarginLoss
import numpy as np
from itertools import cycle
import copy
from torch_geometric.loader import ClusterData, ClusterLoader
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm

def train_supervised(
        model: nn.Module,
        train_loader: ClusterLoader,
        optimizer: optim.Optimizer,
        device: torch.device
        ) -> None:
    r"""
    Trains a model in a supervised manner on a single graph.
    """
    
    model.train()
    ce = nn.CrossEntropyLoss()
    progress_bar = tqdm(train_loader, desc="Training")
    