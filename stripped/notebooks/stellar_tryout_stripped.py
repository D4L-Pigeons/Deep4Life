#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os

os.chdir("..")


# In[47]:


from src.datasets.stellar_data import StellarDataloader, make_graph_list
import random
import numpy as np
import src.datasets.load_d4ls as load_d4ls
import torch
from src.models.vanilla_stellar import VanillaStellarReduced
import pandas as pd


# In[12]:


make_graph_list([], "graphs.pt", 50)


# In[15]:


data_path = load_d4ls.TRAIN_DATA_PATH
file_path = data_path / "graphs.pt"
assert file_path.exists(), f"File graphs.pt does not exist in {data_path}"
graphs = torch.load(file_path)


# In[50]:


sizes = pd.Series([len(g.y) for g in graphs])


# In[51]:


sizes.describe()


# In[25]:


test_idx = np.random.choice(np.arange(125), size=10, replace=False)


# In[26]:


test_mask = np.zeros(125, dtype=bool)
test_mask[test_idx] = True
train_idx = np.where(~test_mask)[0]


# In[38]:


train_data_loader = StellarDataloader(
    "graphs.pt", test=False, batch_size=1, shuffle=True, graphs_idx=train_idx
)
test_data_loader = StellarDataloader(
    "graphs.pt", test=False, batch_size=1, shuffle=False, graphs_idx=test_idx
)


# In[33]:


class ModelConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = 40
        self.hid_dim = 40 * 4
        self.num_classes = 14
        self.lr = 0.001


# Example usage:
config = ModelConfig()
print(config.__dict__)  # Output: 0.001


# In[41]:


stellar = VanillaStellarReduced(config)


# In[44]:


stellar.train(train_data_loader, test_data_loader, epochs=10, return_valid_acc=True)
