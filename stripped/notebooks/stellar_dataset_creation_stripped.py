#!/usr/bin/env python
# coding: utf-8

<<<<<<< HEAD
# In[1]:


import os
if os.getcwd().endswith('notebooks'):
    os.chdir(os.path.dirname(os.getcwd()))

from src.datasets import stellar_data
from src.datasets.stellar_data import make_graph_list, StellarDataloader
from src.datasets import load_d4ls
import matplotlib.pyplot as plt
import numpy as np
=======
# In[14]:


import os

os.chdir("../src")
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import stellar_data
from datasets.stellar_data import make_graph_list_from_anndata, StellarDataloader
from datasets import data_utils
>>>>>>> master


# ## Finding the distances threshold for the cell pairs within a sample
# In the paper the distance threshold is chosen so that there are on average 5 vertices connected to each node. Given the cdf of the distances between cell pairs (with redundant info about distance of pairs) the cells $\mathcal{F}(d)$ the task is to choose $d$, such that
# $$\frac{\mathcal{F}(d)\cdot n_{distances}}{n_{cells}}\approx 5 \rightarrow d \approx \mathcal{F}^{-1}\left(\frac{5\cdot n_{cells}}{n_{distances}}\right)$$
# In the paper clustering bases sampling is used further limiting the number of neighbours in practice, but this is disregarded here.

<<<<<<< HEAD
# In[2]:
=======
# In[4]:
>>>>>>> master


all_distances = stellar_data.get_all_distances()


<<<<<<< HEAD
# In[3]:


plt.hist(all_distances, bins=100, label='Histogram of cell distances')
plt.show()


# In[4]:


n_cells = len(load_d4ls.load_full_anndata())
=======
# In[5]:


plt.hist(all_distances, bins=100)
plt.show()


# In[12]:


n_cells = len(data_utils.load_full_anndata())
>>>>>>> master
n_distances = len(all_distances)
p = 5 * n_cells / n_distances
print(f"The calculated quantile order equals {p}")


<<<<<<< HEAD
# In[5]:
=======
# In[7]:
>>>>>>> master


distance_threshold = np.quantile(all_distances, p)


<<<<<<< HEAD
# In[6]:
=======
# In[8]:
>>>>>>> master


print(f"The calculated distance threshold equals {distance_threshold}")


<<<<<<< HEAD
# In[7]:
=======
# In[9]:
>>>>>>> master


(all_distances < distance_threshold).sum() / n_cells


<<<<<<< HEAD
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
=======
# ## Creating a dataset based on the configuration file

# In[10]:


with open("config/stellar/standard.yaml", "r") as file:
    cfg = argparse.Namespace(**yaml.safe_load(file))

print(cfg)


# In[11]:


cfg.target_labels


# In[13]:


anndata = data_utils.load_full_anndata(test=False)
>>>>>>> master


# In[15]:


<<<<<<< HEAD
data.y


# In[16]:


data.cell_ids

=======
label_encoder = LabelEncoder().fit(cfg.target_labels)


# In[18]:


graph_list = stellar_data.make_graph_list_from_anndata(
    anndata=anndata,
    label_encoder=label_encoder,
    distance_threshold=cfg.distance_threshold,
)


# In[20]:


graph_list[0]


# ### Creating dataloaders based on the created list of graphs

# In[22]:


stellar_dataloader = StellarDataloader(
    graph_list, batch_size=cfg.batch_size, shuffle=True
)


# In[23]:


next(iter(stellar_dataloader))
>>>>>>> master
