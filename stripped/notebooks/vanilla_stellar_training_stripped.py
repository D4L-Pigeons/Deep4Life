#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
if os.getcwd().endswith('notebooks'):
    os.chdir(os.path.dirname(os.getcwd()))
    
from src.datasets import stellar_data
from src.models import vanilla_stellar
import torch
import numpy as np
import argparse


# In[2]:


GRAPH_DATASET_FILENAME = 'stellar_graph_dataset.pt'


# In[3]:


torch.manual_seed(42)
sample_idx_permutations = []
n_folds = 5
n_train = 100
for i in range(n_folds):
    idx_perm = torch.randperm(125).tolist()
    sample_idx_permutations.append({"train" : idx_perm[:n_train], "valid" : idx_perm[n_train:]})


# In[7]:


cfg_reduced = argparse.Namespace(**{
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'input_dim': 40,
    'hid_dim': 128, # originally 128
    'num_classes': 14,
    'lr': 1e-3,
})


# In[8]:


batch_size = 1
n_epochs = 20


# In[9]:


valid_accs = np.zeros(n_folds)
for k in range(n_folds):
    print(f"Fold {k} of {n_folds}")
    train_dataloader = stellar_data.StellarDataloader(
        filename=GRAPH_DATASET_FILENAME,
        batch_size=batch_size,
        shuffle=True,
        graphs_idx=sample_idx_permutations[k]["train"],
        test=False
    )
    valid_dataloader = stellar_data.StellarDataloader(
        filename=GRAPH_DATASET_FILENAME,
        batch_size=batch_size,
        shuffle=False,
        graphs_idx=sample_idx_permutations[k]["valid"],
        test=False
    )

    vanilla_stellar_reduced = vanilla_stellar.VanillaStellarReduced(cfg_reduced)
    valid_acc = vanilla_stellar_reduced.train(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        epochs=n_epochs,
        return_valid_acc=True)
    valid_accs[k] = valid_acc


# In[10]:


print(f"Mean validation accuracy: {valid_accs.mean()} +/- {valid_accs.std()}")

