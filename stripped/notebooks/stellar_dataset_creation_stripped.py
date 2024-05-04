#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
if os.getcwd().endswith('notebooks'):
    os.chdir(os.path.dirname(os.getcwd()))

from src.datasets import stellar_data
from src.datasets.stellar_data import make_graph_list, StellarDataloader
from src.datasets import load_d4ls
import matplotlib.pyplot as plt
import numpy as np


# ## Finding the distances threshold for the cell pairs within a sample
# In the paper the distance threshold is chosen so that there are on average 5 vertices connected to each node. Given the cdf of the distances between cell pairs (with redundant info about distance of pairs) the cells $\mathcal{F}(d)$ the task is to choose $d$, such that
# $$\frac{\mathcal{F}(d)\cdot n_{distances}}{n_{cells}}\approx 5 \rightarrow d \approx \mathcal{F}^{-1}\left(\frac{5\cdot n_{cells}}{n_{distances}}\right)$$
# In the paper clustering bases sampling is used further limiting the number of neighbours in practice, but this is disregarded here.

# In[2]:


all_distances = stellar_data.get_all_distances()


# In[3]:


plt.hist(all_distances, bins=100, label='Histogram of cell distances')
plt.show()


# In[4]:


n_cells = len(load_d4ls.load_full_anndata())
n_distances = len(all_distances)
p = 5 * n_cells / n_distances
print(f"The calculated quantile order equals {p}")


# In[5]:


distance_threshold = np.quantile(all_distances, p)


# In[6]:


print(f"The calculated distance threshold equals {distance_threshold}")


# In[7]:


(all_distances < distance_threshold).sum() / n_cells


# ## Creating and saving a dataset based on the provided threshold

# In[9]:


stellar_data.make_graph_list(
    obs_feature_names=[],
    save_filename='stellar_graph_dataset.pt',
    distance_threshold=distance_threshold,
    test=False
)


# ### Creating dataloaders based on the created list of graphs

# In[10]:


train_dataloader = stellar_data.StellarDataloader(
    filename='stellar_graph_dataset.pt',
    batch_size=1,
    shuffle=True,
    graphs_idx=list(range(100)),
    test=False
)
valid_dataloader = stellar_data.StellarDataloader(
    filename='stellar_graph_dataset.pt',
    batch_size=1,
    shuffle=False,
    graphs_idx=list(range(100, 125)),
    test=False
)


# In[11]:


data = next(iter(train_dataloader))


# In[15]:


data.y


# In[16]:


data.cell_ids

