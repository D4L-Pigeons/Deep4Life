#!/usr/bin/env python
# coding: utf-8

# ## Setup

# ## Run this if running on colab

# In[8]:


from IPython.display import clear_output; token = input(); clear_output()


# In[9]:


get_ipython().system(' git clone https://$token@github.com/SzymonLukasik/Deep4Life.git')


# In[10]:


get_ipython().run_line_magic('cd', '/content/Deep4Life')


# In[11]:


get_ipython().system('pip install anndata')


# In[ ]:


get_ipython().system(' pip install pyometiff')


# ## Imports

# In[12]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pyometiff
import os
import gdown
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV

from typing import List
from src.datasets import load_d4ls
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


get_ipython().system('mkdir data')


# In[ ]:


get_ipython().system('gdown 1-0YOHE1VoTRWqfBJLHQorGcHmkhCYvqW')


# In[7]:


load_d4ls.DATA_PATH


# In[ ]:


get_ipython().system('unzip train.zip -d $load_d4ls.DATA_PATH')


# ## Load anndata

# In[13]:


train_anndata = load_d4ls.load_full_anndata()


# # SVM Baseline

# ### Dataset Prep

# In[2]:


np.random.seed(42)


# In[3]:


def get_edge_index(pos, sample_ids, distance_thres):
    # construct edge indexes when there is region information
    edge_list = []
    sample_ids_unique = np.unique(sample_ids)
    for sample_id in sample_ids_unique:
        locs = np.where(sample_ids == sample_id)[0]
        pos_region = pos[locs, :]
        dists = pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for i, j in region_edge_list:
            edge_list.append([locs[i], locs[j]])
    return edge_list


# In[4]:


def get_train_test_masks(train_anndata, test_count=0):
    sample_ids = train_anndata.obs["sample_id"]
    sample_ids_unique = np.unique(sample_ids)

    sample_ids_idx = np.random.choice(np.arange(len(sample_ids_unique)), test_count, replace=False)
    test_sample_ids_mask = np.zeros_like(sample_ids_unique, dtype=bool)
    test_sample_ids_mask[sample_ids_idx] = True

    test_unique_sample_ids = sample_ids_unique[test_sample_ids_mask]

    test_mask = sample_ids.isin(test_unique_sample_ids)
    train_mask = ~test_mask

    return train_mask, test_mask


# In[5]:


def prepare_data(train_anndata, make_graph=False, test_samples=10):
    train_mask, test_mask = get_train_test_masks(train_anndata, test_samples)

    X = train_anndata.layers['exprs']
    X_train = X[train_mask]
    X_test = X[test_mask]

    pos = train_anndata.obs[["Pos_X", "Pos_Y"]].values
    pos_train = pos[train_mask]
    pos_test = pos[test_mask]

    if make_graph:
        sample_ids = train_anndata.obs["sample_id"]
        test_sample_ids = sample_ids[test_mask]
        train_sample_ids = sample_ids[train_mask]

        edges_train = get_edge_index(pos_train, train_sample_ids, 10)
        edges_test = get_edge_index(pos_test, test_sample_ids, 10)
    else:
        edges_train = None
        edges_test = None

    cell_types = np.sort(list(set(train_anndata.obs["cell_labels"].values))).tolist()
    # we here map class in texts to categorical numbers and also save an inverse_dict to map the numbers back to texts
    cell_type_dict = {}
    inverse_dict = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type

    Y_train = train_anndata.obs["cell_labels"].values[train_mask]
    Y_test = train_anndata.obs["cell_labels"].values[test_mask]

    Y_train = np.array([cell_type_dict[x] for x in Y_train])
    Y_test = np.array([cell_type_dict[x] for x in Y_test])

    return X_train, Y_train, edges_train, X_test, Y_test, edges_test, inverse_dict



# In[14]:


X_train, Y_train, edges_train, X_test, Y_test, edges_test, inverse_dict = prepare_data(train_anndata)


# In[15]:


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)


# In[19]:


train_anndata.obs.head()


# In[20]:


train_anndata.var


# In[21]:


X_train[:1, :]


# In[22]:


X_train_scaled[:1, :]


# In[23]:


X_train.shape


# ## Grid Search on sampled dataset

# In[15]:


n_samples = 10_000

sample_perm = np.random.permutation(np.arange(X_train.shape[0]))[:n_samples]
X_train_sampled = X_train[sample_perm]
Y_train_sampled = Y_train[sample_perm]


# In[ ]:


LinearSVC_param_grid = {
    'penalty': ['l1', 'l2'],
    'dual': [False],
    'C': [0.1, 0.3, 0.5, 0.7, 0.9],
}

linear_svc_grid_search = GridSearchCV(LinearSVC(), LinearSVC_param_grid, verbose=1, n_jobs=-1)
linear_svc_grid_search.fit(X_train_sampled, Y_train_sampled)


# In[ ]:


pd.DataFrame(linear_svc_grid_search.cv_results_)


# In[ ]:


SVC_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
    }


svc_grid_search = GridSearchCV(SVC(), SVC_param_grid, refit = True, verbose = 1)
svc_grid_search.fit(X_train_sampled, Y_train_sampled)


# In[ ]:


pd.DataFrame(svc_grid_search.cv_results_)


# In[16]:


svc_best_configuration = {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
svc = SVC(**svc_best_configuration)
svc.fit(X_train_scaled, Y_train)


# In[ ]:


svc.score(X_train_scaled, Y_train)


# In[17]:


svc.score(X_test_scaled, Y_test)


# In[ ]:


# fit an svm model

SVC_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear']}
# C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
# gamma is the parameter of the RBF kernel and can be thought of as the ‘spread’ of the kernel and therefore the decision region.
# kernel is the type of kernel used in the algorithm. The most common ones are ‘linear’, ‘poly’, and ‘rbf’.

SVC_grid = GridSearchCV(SVC(), SVC_param_grid, refit = True, verbose = 1)
SVC_grid.fit(X_train_scaled[:trim], Y_train[:trim])


# In[ ]:


trim = 50_000
svc = SVC(C=1, gamma=0.1, kernel='linear', verbose=True)
svc.fit(X_train_scaled[:trim], Y_train[:trim])


# In[ ]:


svc.score(X_train_scaled[trim:], Y_train[trim:])


# In[ ]:


svc.score(X_train_scaled, Y_train)


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


svc.score(X_test_scaled, Y_test)


# ## Fit on the full dataset

# In[ ]:


trim = len(X_train)
svc_larger = SVC(C=1, gamma=0.1, kernel='linear', verbose=True)
svc_larger.fit(X_train_scaled[:trim], Y_train[:trim])


# In[ ]:


svc_larger.score(X_train_scaled[trim:], Y_train[trim:])


# In[ ]:


svc_larger.score(X_train_scaled, Y_train)


# In[ ]:


svc_larger.score(X_test, Y_test)


# In[ ]:


svc_larger.score(X_test_scaled, Y_test)

