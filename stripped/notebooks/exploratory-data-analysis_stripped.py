#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[15]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import List

from datasets import load_d4ls

pd.set_option('display.max_columns', None)


# ## EDA utils

# In[121]:


def make_class_distribution_plot(
        df: DataFrame,
        class_var: str,
        title: str,
        ) -> None:
    r"""
    Plot the distribution of the given class variable in the dataframe.
    """

    plt.figure(figsize=(10, 4))
    plt.bar(df[class_var].value_counts().index, df[class_var].value_counts().values / len(df), color='grey', edgecolor='black')
    plt.xticks(rotation=35)
    plt.xlabel(class_var)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def make_histograms(
        df: DataFrame,
        quant_vars: List[str],
        title: str,
        ) -> None:
    r"""
    Plot histograms of the given quantitative variables in the dataframe.
    """

    num_plots = len(quant_vars)
    num_rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
    fig.suptitle(title)

    for i, var in enumerate(quant_vars):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.hist(df[var], density=True, bins=100, color='grey', edgecolor='black')
        ax.set_xlabel(var)
        ax.set_ylabel('Probability Density')

    plt.show()


def make_boxplots(
        df: DataFrame,
        quant_vars: List[str],
        title: str,
        ) -> None:
    r"""
    Plot boxplots of the given quantitative variables in the dataframe.
    """

    num_plots = len(quant_vars)
    num_rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    fig.suptitle(title)

    for i, var in enumerate(quant_vars):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.boxplot(df[var], vert=False, patch_artist=True, boxprops=dict(facecolor='grey', color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=3, linestyle='none'))
        ax.set_xlabel(var)
        ax.set_ylabel('Value')

    plt.show()


def make_boxplots_per_celltype(
        df: DataFrame,
        class_var: str,
        quant_var: str,
        title: str,
        ) -> None:
    r"""
    Plot boxplots of the given quantitative variables in the dataframe, grouped by cell type.
    """

    x = [group[quant_var].values for _, group in df.groupby(class_var, observed=True)]
    labels = df[class_var].unique()

    plt.figure(figsize=(12, 6))
    plt.boxplot(x=x, labels=labels, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='grey', color='black'), 
                whiskerprops=dict(color='black'), 
                capprops=dict(color='black'), 
                medianprops=dict(color='black'), 
                flierprops=dict(marker='o', markersize=3, linestyle='none'))
    plt.title(title)
    plt.show()


def make_corr_plot(
        df: DataFrame,
        vars: List[str],
        title: str,
        method: str = 'spearman',
        ) -> None:
    r"""
    Plot a correlation matrix of the given variables in the dataframe.
    """

    correlation_matrix = df[vars].corr(method=method)
    plt.figure(figsize=(5, 4))
    plt.imshow(correlation_matrix, cmap='grey', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(vars)), vars, rotation=45)
    plt.yticks(range(len(vars)), vars)
    plt.title(title)
    plt.show()


# ## Loading anndata

# In[7]:


train_anndata = load_d4ls.load_full_anndata()


# ## Tabular data exploration

# ### Checking types and missing data

# In[1]:


display(train_anndata.obs.columns)


# In[ ]:


display(train_anndata.obs.dtypes)


# In[ ]:


display(train_anndata.obs.head())


# In[ ]:


display(train_anndata.obs.isna().sum())


# `train_anndata.var` stores the information about the marker names. Please use the `marker` column in your analysis. `use_channel` indicates wheather a channel is used in practice.

# In[9]:


display(train_anndata.var.columns)
display(train_anndata.var.dtypes)
display(train_anndata.var.head())
display(train_anndata.var.isna().sum())


# `train_anndata.layers['exprs']` with shape `(train_anndata.obs.shape[0], train_anndata.var.shape[0])` stores the matrix with marker expressions for each cell:

# In[10]:


display(train_anndata.layers)
layer_types = [type(layer) for layer in train_anndata.layers.values()]
display(layer_types)
display(train_anndata.layers['counts'].shape)
display(train_anndata.layers['counts'].dtype)
display(train_anndata.layers['exprs'].shape)
display(train_anndata.layers['exprs'].dtype)
display(train_anndata.layers['counts'][0, :])
display(train_anndata.layers['exprs'][0, :])


# ### Checking values

# In[11]:


train_anndata.obs.columns


# Train `anndata` `obs` dataframe stores the information about cells. Each row in this table represent an information about a single cell. It has the following columns that are interesting for your analysis:
# - `image` - name of the image file from which a cell was obtained,
# - `sample_id` - name of the patient sample from which a given image was obtained,
# - `ObjectNumber` - a cell number within a given image (note that it starts from 1),
# - `Pos_X`, `Pos_Y` - a spatial position of the cell one the image,
# - `area`, `major_axis_length`, `minor_axis_length`, `eccentricity`, `width_px`, `height_px` - shape-derived features of a cell,
# - `Batch` - a batch in which a sample was used,
# - `cell_labels` - your target cell type annotations. **THIS IS WHAT YOUR MODELS AIM TO PREDICT!**

# ### Cell type distribution

# In[122]:


make_class_distribution_plot(train_anndata.obs, 'celltypes', 'Class Distribution')


# ### Independent obs variables distributions

# In[123]:


obs_quant_vars = ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'width_px', 'height_px']
train_anndata.obs[obs_quant_vars].describe()


# In[124]:


make_histograms(train_anndata.obs, obs_quant_vars, 'Histograms of Quantitative Variables in the obs Training Data')


# In[125]:


make_boxplots(train_anndata.obs, obs_quant_vars, 'Boxplots of Quantitative Variables in the obs Training Data')


# ### Independent obs variables distributions per cell type

# In[126]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'area', 'Boxplots of Area by Cell Type in the obs Training Data')
make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'major_axis_length', 'Boxplots of Major Axis Length by Cell Type in the obs Training Data')
make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'minor_axis_length', 'Boxplots of Minor Axis Length by Cell Type in the obs Training Data')
make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'eccentricity', 'Boxplots of Eccentricity by Cell Type in the obs Training Data')
make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'width_px', 'Boxplots of Width by Cell Type in the obs Training Data')
make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'height_px', 'Boxplots of Height by Cell Type in the obs Training Data')


# ### Independent obs variables correlations

# In[127]:


make_corr_plot(train_anndata.obs, obs_quant_vars, 'Correlation Matrix of Quantitative Variables in the obs Training Data (spearman)', method='spearman')
make_corr_plot(train_anndata.obs, obs_quant_vars, 'Correlation Matrix of Quantitative Variables in the obs Training Data (pearson)', method='pearson')


# ### Independent gene expression variables distributions

# In[145]:


train_anndata.var['marker'].values


# In[143]:


plt.imshow(np.corrcoef(train_anndata.layers['exprs'].T))


# In[147]:


marker_names = train_anndata.var['marker'].values

df = pd.DataFrame(values=train_anndata.layers['exprs'], columns=marker_names)
make_corr_plot(df, marker_names, 'Correlation Matrix of Marker Expression in the Training Data (spearman)', method='spearman')


# ### PCA 2 components plot

# ### Umap

# ### T-sne

# ### Self organizing maps

# 
