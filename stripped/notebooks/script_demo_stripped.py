#!/usr/bin/env python
# coding: utf-8

# ## Setup

# ## Run this if running on colab

# In[ ]:


from IPython.display import clear_output; token = input(); clear_output()


# In[ ]:


get_ipython().system(' git clone https://$token@github.com/SzymonLukasik/Deep4Life.git')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/Deep4Life')


# In[ ]:


get_ipython().system('pip install anndata')


# In[ ]:


get_ipython().system(' pip install pyometiff')


# ## Imports

# In[ ]:


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


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().system('mkdir data')


# In[ ]:


get_ipython().system('gdown 1-0YOHE1VoTRWqfBJLHQorGcHmkhCYvqW')


# In[ ]:


load_d4ls.DATA_PATH


# In[ ]:


get_ipython().system('unzip train.zip -d $load_d4ls.DATA_PATH')


# ## Load anndata

# In[ ]:


train_anndata = load_d4ls.load_full_anndata()


# # Training from the command line

# In[ ]:


get_ipython().system(' git checkout lukass/svm_baseline')


# In[7]:


get_ipython().system(' git pull origin')


# In[19]:


get_ipython().system(' git status')


# In[15]:


get_ipython().system(' git checkout -- src/models/sklearn_svm.py')


# In[20]:


get_ipython().system(' git branch')


# In[ ]:


get_ipython().system(' pip install scanpy')


# In[ ]:


get_ipython().system(' pip install scikit_learn==1.4.2')


# ## SVM Baseline

# In[18]:


get_ipython().system('python3 src/train_and_validate.py --method sklearn_svm/svc --config linear')


# In[ ]:


get_ipython().system('python3 src/train_and_validate.py --method sklearn_svm/svc --config linear')


# ##  Other methods

# In[274]:


get_ipython().system(' git fetch origin master')


# In[24]:


get_ipython().system('python3 src/train_and_validate.py --method xgboost --config standard')


# In[25]:


get_ipython().system('python3 src/train_and_validate.py --method sklearn_mlp --config standard')


# In[26]:


get_ipython().system('python3 src/train_and_validate.py --method torch_mlp --config standard')


# In[27]:


get_ipython().system('python3 src/train_and_validate.py --method stellar --config standard')


# In[28]:


get_ipython().system('python3 src/train_and_validate.py --method stellar --config custom')


# In[29]:


get_ipython().system('python3 src/train_and_validate.py --method stellar --config custom_random_nodes')


# In[34]:


get_ipython().system('pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html')


# In[35]:


get_ipython().system('python3 src/train_and_validate.py --method stellar --config custom_neighbors')


# # Test

# In[277]:


get_ipython().system('git fetch origin')


# In[278]:


get_ipython().system('git pull origin')


# In[54]:


get_ipython().system('git checkout lukass/introduce_test_mode')


# ## Creating a dummy test set

# In[ ]:


from src.datasets.data_utils import load_full_anndata
test_anndata = load_full_anndata(False)


# In[265]:


import scanpy

sampled_anndata = scanpy.pp.subsample(test_anndata, n_obs=1000, copy=True, random_state=42)
sampled_anndata.write_h5ad("/content/Deep4Life/data/test/cell_data.h5ad")
len(sampled_anndata.obs["cell_labels"].cat.categories)


# In[ ]:


filenames_list = sampled_anndata.obs["image"].unique()
len(filenames_list)


# In[ ]:


filenames = "
".join(filenames_list)
with open("filenames", "w") as f:
  f.write(filenames)


# In[254]:


get_ipython().system(' ls data/train/images_masks/img | wc -l')


# In[256]:


get_ipython().system(' mkdir -p data/test/images_masks/img')


# In[257]:


get_ipython().system(' mkdir -p data/test/images_masks/masks')


# In[259]:


get_ipython().system('cat filenames|wc -l')


# In[260]:


get_ipython().system('cat filenames | xargs -I {} cp data/train/images_masks/img/{} data/test/images_masks/img')


# In[261]:


get_ipython().system('cat filenames | xargs -I {} cp data/train/images_masks/masks/{} data/test/images_masks/masks')


# In[262]:


get_ipython().system(' ls data/test/images_masks/img | wc -l')


# In[263]:


get_ipython().system(' ls data/test/images_masks/masks | wc -l')


# ## Running inference on a trained svm

# In[272]:


get_ipython().system('python3 src/train_and_validate.py --method sklearn_svm/svc --config linear test linear_2024-05-08_20-33-44_seed_42_folds_5')


# ## Running on best results from the drive (I created a shurtcut to my drive to the shared folder)

# In[270]:


get_ipython().system('cp -r /content/drive/MyDrive/best_results/* ./results/')


# In[279]:


get_ipython().system('python3 src/train_and_validate.py --method stellar --config custom test custom_2024-05-08_22-23-22_seed_42_folds_5')


# In[281]:


get_ipython().system('python3 src/train_and_validate.py --method xgboost --config standard test standard_2024-05-08_21-15-37_seed_42_folds_5')

