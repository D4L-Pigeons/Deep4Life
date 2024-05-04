#!/usr/bin/env python
# coding: utf-8

# ## 0. Setup

# ### Imports

# In[4]:


import os
if os.getcwd().endswith('notebooks'):
    os.chdir(os.path.dirname(os.getcwd()))

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import FactorAnalysis, PCA, KernelPCA, FastICA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import List, Tuple, Optional

from src.datasets import load_d4ls

pd.set_option('display.max_columns', None)


# ### EDA utils

# In[109]:


def make_class_distribution_plot(
        df: DataFrame,
        class_var: str,
        title: str,
        display_desc: bool = False
        ) -> None:
    r"""
    Plot the distribution of the given class variable in the dataframe.
    """

    plt.figure(figsize=(10, 4))
    plt.bar(df[class_var].value_counts().index, df[class_var].value_counts().values / len(df), color='grey', edgecolor='black')
    plt.xticks(rotation=35)
    plt.ylabel('Frequency')
    if display_desc:
        plt.xlabel(class_var)
        plt.title(title)
    plt.show()

def make_histograms(
        df: DataFrame,
        quant_vars: List[str],
        title: str,
        subplots_shape: Optional[Tuple] = None,
        display_desc: bool = False,
        title_shift: float = 1
        ) -> None:
    r"""
    Plot histograms of the given quantitative variables in the dataframe.
    """

    num_plots = len(quant_vars)
    if subplots_shape is not None:
        num_rows, num_cols = subplots_shape
    else:
        num_rows = (num_plots + 1) // 2
        num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 2 * num_rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)    
    
    if display_desc:
        fig.suptitle(title, fontsize=16, y=title_shift)

    for i, var in enumerate(quant_vars):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.hist(df[var], density=True, bins=100, color='grey', edgecolor='black')
        ax.set_xlabel(var)
        ax.set_ylabel('PDF estimation', fontsize=7)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  


    plt.show()


def make_boxplots(
        df: DataFrame,
        quant_vars: List[str],
        title: str,
        subplots_shape: Optional[Tuple] = None,
        display_desc: bool = False,
        title_shift: float = 1
        ) -> None:
    r"""
    Plot boxplots of the given quantitative variables in the dataframe.
    """

    num_plots = len(quant_vars)
    if subplots_shape is not None:
        num_rows, num_cols = subplots_shape
    else:
        num_rows = (num_plots + 1) // 2
        num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 2 * num_rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)    
    
    if display_desc:
        fig.suptitle(title, fontsize=16, y=title_shift)

    for i, var in enumerate(quant_vars):
        row = i // num_cols
        col = i % num_cols
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
        display_desc: bool = False
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
    
    if display_desc:
        plt.title(title)
    
    plt.show()


def make_corr_plot(
        df: DataFrame,
        vars: List[str],
        title: str,
        method: str = 'spearman'
        ) -> None:
    r"""
    Plot a correlation matrix of the given variables in the dataframe.
    """

    correlation_matrix = df[vars].corr(method=method)
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(255, 10, as_cmap=True)
    plt.imshow(correlation_matrix, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(vars)), vars, rotation=90)
    plt.yticks(range(len(vars)), vars)
    plt.title(title)
    plt.show()


# ### Loading anndata

# In[3]:


train_anndata = load_d4ls.load_full_anndata()
marker_names = train_anndata.var['marker'].values
df = pd.DataFrame(data=train_anndata.layers['exprs'], columns=marker_names)
targets_np = train_anndata.obs['cell_labels'].values.to_numpy()
df['cell_labels'] = targets_np


# ### Tabular data exploration

# #### Checking columns, data types and missing data in the `train_anndata.obs`, wchich stores cell features other than marker expressions
# 
# Train `anndata` `obs` dataframe stores the information about cells. Each row in this table represent an information about a single cell. It has the following columns that are interesting for your analysis:
# - `image` - name of the image file from which a cell was obtained,
# - `sample_id` - name of the patient sample from which a given image was obtained,
# - `ObjectNumber` - a cell number within a given image (note that it starts from 1),
# - `Pos_X`, `Pos_Y` - a spatial position of the cell one the image,
# - `area`, `major_axis_length`, `minor_axis_length`, `eccentricity`, `width_px`, `height_px` - shape-derived features of a cell,
# - `Batch` - a batch in which a sample was used,
# - `cell_labels` - your target cell type annotations. **THIS IS WHAT YOUR MODELS AIM TO PREDICT!**

# In[4]:


display(train_anndata.obs.columns)


# In[5]:


display(train_anndata.obs.dtypes)


# In[7]:


display(train_anndata.obs.isna().sum())


# #### Viewing the `train_anndata.obs` dataset

# In[6]:


display(train_anndata.obs.head())


# #### Checking columns, data types and missing data in the `train_anndata.var`, which stores the information about the marker names.
# `use_channel` indicates wheather a channel is used in practice.

# In[44]:


display(train_anndata.var.columns)


# In[45]:


display(train_anndata.var.dtypes)


# In[46]:


display(train_anndata.var.isna().sum())


# #### Viewing the `train_anndata.var` dataset

# In[47]:


display(train_anndata.var.head())


# #### Checking keys, data types in the `train_anndata.layers`, which stores the information about the marker expressions.
# 
# `train_anndata.layers['exprs']` with shape `(train_anndata.obs.shape[0], train_anndata.var.shape[0])` stores the matrix with marker expressions for each cell

# In[48]:


display(train_anndata.layers)


# In[49]:


layer_types = [type(layer) for layer in train_anndata.layers.values()]
display(layer_types)


# In[50]:


display(train_anndata.layers['counts'].shape)
display(train_anndata.layers['counts'].dtype)


# In[51]:


display(train_anndata.layers['exprs'].shape)
display(train_anndata.layers['exprs'].dtype)


# #### Viewing the `train_anndata.layers['counts']` and `train_anndata.layers['exprs']`

# In[52]:


display(train_anndata.layers['counts'][0, :])
display(train_anndata.layers['exprs'][0, :])


# #### Checking values

# In[10]:


train_anndata.obs.columns


# ###

# ## 1. General data overview

# ### Cell type

# In[78]:


train_anndata.obs.cell_labels.unique().tolist()


# #### Short introduction of the cell types
# 1. **MacCD163 (Macrophage CD163)**: Macrophages are immune cells that play a crucial role in tissue homeostasis, inflammation, and immune defense. CD163 is a marker expressed on macrophages, particularly those involved in scavenging damaged cells and tissues.
# 
# 2. **Mural (Mural cells)**: Mural cells typically refer to the supportive cells found in the walls of blood vessels, including pericytes and smooth muscle cells. They regulate blood vessel tone, stability, and permeability.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are professional antigen-presenting cells that play a key role in initiating and regulating immune responses. They capture antigens, process them, and present them to T cells, thereby initiating adaptive immune responses.
# 
# 4. **Tumor**: Tumor cells are abnormal cells that proliferate uncontrollably, leading to the formation of a mass or tumor. They can originate from various tissues and can have different properties depending on their type and stage.
# 
# 5. **CD4 (CD4+ T cell)**: CD4+ T cells, also known as helper T cells, are a type of lymphocyte that coordinates immune responses by secreting cytokines and interacting with other immune cells. They are crucial for activating other immune cells, including CD8+ T cells and B cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: HLA-DR is a major histocompatibility complex (MHC) class II molecule expressed on antigen-presenting cells such as dendritic cells, macrophages, and B cells. It plays a key role in presenting antigens to CD4+ T cells and initiating adaptive immune responses.
# 
# 7. **NK (Natural Killer) cell**: Natural Killer cells are a type of cytotoxic lymphocyte that plays a critical role in the innate immune response against virus-infected cells and tumor cells. They can directly kill target cells without prior sensitization.
# 
# 8. **CD8 (CD8+ T cell)**: CD8+ T cells, also known as cytotoxic T cells, are a type of lymphocyte that directly kills infected or abnormal cells. They recognize antigens presented on MHC class I molecules and induce apoptosis in target cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are a subset of CD4+ T cells that suppress immune responses, thereby maintaining immune tolerance and preventing autoimmunity. They play a crucial role in regulating immune homeostasis and preventing excessive inflammation.
# 
# 10. **Neutrophil**: Neutrophils are the most abundant type of white blood cells and are essential for the innate immune response against bacterial and fungal infections. They migrate to sites of infection and phagocytose pathogens.
# 
# 11. **Plasma cell**: Plasma cells are terminally differentiated B cells that produce and secrete large amounts of antibodies (immunoglobulins). They are key effectors of the humoral immune response and provide long-term immunity against pathogens.
# 
# 12. **B cell**: B cells are a type of lymphocyte that plays a central role in the adaptive immune response by producing antibodies and presenting antigens to T cells. They mature into plasma cells upon activation.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are a specialized subset of dendritic cells that are particularly adept at producing type I interferons in response to viral infections. They play a crucial role in antiviral immunity.
# 
# 14. **BnT**: This term is not standard in cell biology, so it's difficult to provide a specific introduction without further context. It might refer to a hybrid cell type combining characteristics of B and T cells, but clarification would be needed to provide a detailed description.

# In[53]:


train_anndata.obs.cell_labels.value_counts()


# In[70]:


make_class_distribution_plot(train_anndata.obs, 'cell_labels', 'Figure 1.1 Class Distribution', display_desc=True)


# #### Interpretation
# As visible on the Figure 1 the cell labels distribution is highly unbalanced with half of the examples belonging to the Tumor class. This issue shold be taken into consideration in the trianing process if it is desired for each class to contribute "equally" in the training process.

# ### Independent obs variables distributions

# In[72]:


obs_quant_vars = ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'width_px', 'height_px']
train_anndata.obs[obs_quant_vars].describe()


# In[108]:


make_histograms(train_anndata.obs, obs_quant_vars, 'Figure 1.2 Histograms of Quantitative Variables in the obs Training Data', display_desc=True, title_shift=0.95)


# In[110]:


make_boxplots(train_anndata.obs, obs_quant_vars, 'Figure 1.3 Boxplots of Quantitative Variables in the obs Training Data', display_desc=True, title_shift=0.95)


# #### Interpretation
# Figure 1.2 and Figure 1.3 summarise the distributions of the cell size features. It is visible that the `major_axis_length` and `minor_axis_length` are more informative than `width_px` and `height_px` in visual examination as outliers take more extreme values ing the latter case.

# ### Independent obs variables distributions per cell type

# In[79]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'area', 'Figure 1.4 Boxplots of Area by Cell Type in the obs Training Data', display_desc=True)


# In[81]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'major_axis_length', 'Figure 1.5 Boxplots of Major Axis Length by Cell Type in the obs Training Data', display_desc=True)


# In[86]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'minor_axis_length', 'Figure 1.6 Boxplots of Minor Axis Length by Cell Type in the obs Training Data', display_desc=True)


# In[87]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'eccentricity', 'Figure 1.7 Boxplots of Eccentricity by Cell Type in the obs Training Data', display_desc=True)


# In[84]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'width_px', 'Boxplots of Width by Cell Type in the obs Training Data')


# In[85]:


make_boxplots_per_celltype(train_anndata.obs, 'celltypes', 'height_px', 'Boxplots of Height by Cell Type in the obs Training Data')


# #### Interpretation
# From the Figures 1.4-1.7 it is noticable, that for the undefined cell type the right tail of the distribution is larger than for the rest of the distributions.

# ### Independent marker variables distributions

# In[19]:


df[marker_names].describe()


# In[113]:


make_histograms(df, marker_names, 'Figure 1.8 Histograms of the Markers', subplots_shape=(10, 4), display_desc=True, title_shift=0.9)


# In[115]:


make_boxplots(df, marker_names, 'Figure 1.9 Boxplots of the Markers in the Training Data', subplots_shape=(10, 4), display_desc=True, title_shift=0.9)


# #### Interpretation
# As visible on the Figures 1.8 and 1.9 the distributions are mostly right skewed with exceptions including proportinally more bell shaped DNA1, DNA2, CD33 and CD14 or bimodal Ecad.

# ### Independent markers variables distributions per cell type

# In[117]:


marker_number = 0
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit variable MPO expression depending on their activation state and tissue microenvironment. MPO expression in these macrophages could indicate their involvement in inflammatory processes or tissue remodeling.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with MPO expression. MPO is primarily expressed in cells of myeloid lineage, particularly neutrophils.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not known to express MPO under normal physiological conditions. Their primary function revolves around antigen presentation and immune activation rather than producing enzymes like MPO.
# 
# 4. **Tumor**: MPO expression in tumor cells can vary widely depending on the tumor type. Some leukemias and lymphomas might express MPO, whereas most solid tumors are not expected to express this myeloid marker.
# 
# 5. **CD4**: CD4+ T cells typically do not express MPO. These cells are involved in coordinating immune responses through cytokine secretion rather than producing enzymes like MPO.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not known to express MPO under normal physiological conditions.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not associated with MPO expression. They primarily exert cytotoxic activity against infected or abnormal cells through mechanisms other than MPO.
# 
# 8. **CD8**: CD8+ T cells, also known as cytotoxic T cells, do not express MPO. Their primary function is to recognize and eliminate virus-infected or abnormal cells through the release of cytotoxic molecules like perforin and granzyme.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are unlikely to express MPO, as their role is primarily immunosuppressive rather than directly participating in antimicrobial or inflammatory processes mediated by MPO.
# 
# 10. **Neutrophil**: Neutrophils are the primary cells known for expressing high levels of MPO. MPO is a key enzyme in neutrophils involved in antimicrobial defense through the generation of reactive oxygen species.
# 
# 11. **Plasma**: Plasma cells are differentiated B cells that produce antibodies but do not express MPO. Their primary role is in humoral immunity rather than innate immune responses mediated by enzymes like MPO.
# 
# 12. **B cell**: B cells, including plasma cells, typically do not express MPO. Their main function is antibody production and antigen presentation to T cells rather than enzymatic activity like MPO.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Similar to conventional dendritic cells, plasmacytoid dendritic cells are not known to express MPO. Their function primarily revolves around type I interferon production in response to viral infections.
# 
# 14. **BnT**: Without specific context or clarification regarding the "BnT" designation, it's challenging to infer potential MPO expression patterns in this cell type. If "BnT" refers to a specific cell population, additional information would be needed to assess its MPO expression profile accurately.

# In[118]:


marker_number = 1
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Moderate to high expression of Histone H3 is expected, reflecting its role in gene regulation and chromatin remodeling during immune responses.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, may exhibit stable expression of Histone H3, although levels might be lower compared to actively dividing cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express Histone H3, particularly as they regulate gene expression during immune responses and antigen presentation.
# 
# 4. **Tumor**: Histone H3 expression in tumor cells can vary widely depending on the type and stage of cancer. Alterations in histone modifications are common in cancer, impacting gene expression and tumor progression.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with Histone H3 expression. Their role is more focused on cytokine secretion and immune regulation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and macrophages, may exhibit stable expression patterns of Histone H3 associated with their antigen-presenting functions.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with Histone H3 expression. Their functions are primarily mediated through cytotoxic activity rather than chromatin remodeling.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with Histone H3 expression. Their role is more focused on cytotoxicity and immune surveillance.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are unlikely to express Histone H3, as their function is more related to immune suppression than chromatin remodeling.
# 
# 10. **Neutrophil**: Neutrophils may express Histone H3 as part of their chromatin structure, although levels might decrease as they mature and undergo chromatin decondensation during activation.
# 
# 11. **Plasma**: Plasma cells are not typically associated with Histone H3 expression. Their primary function is antibody production rather than chromatin remodeling.
# 
# 12. **B cell**: B cells, including plasma cells, are not typically associated with Histone H3 expression. Their role is more focused on antibody production and antigen presentation.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with Histone H3 expression. Their functions are more related to antiviral immune responses.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific Histone H3 expression patterns without further context or clarification.
# 
# Overall, Histone H3 expression is essential for maintaining chromatin structure and regulating gene expression across various cell types, but the levels and patterns may vary depending on the cell's activation state, differentiation status, and specific functions.

# In[119]:


marker_number = 2
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with SMA expression as SMA is primarily found in smooth muscle cells and myofibroblasts.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are expected to strongly express SMA as it is a characteristic marker of these cells involved in vascular tone and stability.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with SMA expression as they are part of the immune system and are not involved in muscle-related functions.
# 
# 4. **Tumor**: SMA expression in tumor cells can vary widely depending on the tumor type and microenvironment. In some cases, SMA expression may indicate differentiation toward a myofibroblastic phenotype or involvement of stromal cells expressing SMA.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and macrophages, are not typically associated with SMA expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 11. **Plasma**: Plasma cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 12. **B cell**: B cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with SMA expression as SMA is primarily found in cells of mesenchymal origin rather than immune cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific SMA expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential SMA expression accurately. Without such context, it's difficult to provide a definitive answer regarding SMA expression in BnT cells. If you can provide more details or context about the "BnT" designation, I'd be happy to assist further.

# In[120]:


marker_number = 3
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Moderate to high expression of CD16 may be observed, particularly in activated or inflammatory macrophages involved in immune responses and phagocytosis.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD16 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with CD16 expression. CD16 is more commonly found on cells of the myeloid lineage, such as macrophages and neutrophils.
# 
# 4. **Tumor**: CD16 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD16, such as macrophages or NK cells, may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD16 expression. CD16 is more commonly found on cells of the innate immune system, such as NK cells and macrophages.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with CD16 expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are known to express CD16, which plays a crucial role in their cytotoxic activity through antibody-dependent cellular cytotoxicity (ADCC).
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD16 expression. CD16 is more commonly found on cells of the innate immune system, such as NK cells and macrophages.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD16 expression.
# 
# 10. **Neutrophil**: Neutrophils are known to express CD16, which plays a role in their effector functions, including phagocytosis and ADCC.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD16 expression.
# 
# 12. **B cell**: B cells are not typically associated with CD16 expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD16 expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD16 expression patterns without further context or clarification.
# 
# Overall, CD16 expression is primarily associated with cells of the innate immune system, such as NK cells and neutrophils, rather than cells of the adaptive immune system, such as T cells and B cells. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[121]:


marker_number = 4
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Moderate to high expression of CD38 may be observed, particularly in activated or inflammatory macrophages involved in immune responses and tissue remodeling.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD38 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD38 expression, particularly in mature or activated states involved in antigen presentation and immune regulation.
# 
# 4. **Tumor**: CD38 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD38 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may exhibit CD38 expression, especially upon activation, differentiation into effector subsets, or in regulatory T cell populations.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD38 expression, particularly under inflammatory or immune stimulatory conditions.
# 
# 7. **NK (Natural Killer) cells**: NK cells may express CD38, particularly in activated or cytotoxic subsets involved in immune surveillance and antitumor responses.
# 
# 8. **CD8**: CD8+ T cells may express CD38, particularly in activated or memory subsets associated with cytotoxic activity against infected or malignant cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit CD38 expression, particularly in subsets associated with suppressive functions and immune regulation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD38 expression.
# 
# 11. **Plasma**: Plasma cells may exhibit high levels of CD38 expression, serving as a hallmark marker for their identification and involvement in antibody production.
# 
# 12. **B cell**: B cells, particularly activated or memory subsets, may express CD38, reflecting their involvement in immune responses and antibody production.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit CD38 expression, particularly in response to viral infections or inflammatory stimuli.
# 
# 14. **BnT**: Without further context or clarification on the designation "BnT," it's difficult to assess the potential CD38 expression pattern accurately. If "BnT" refers to a specific cell type or population, additional information would be needed to determine its CD38 expression pattern.
# 
# Overall, CD38 expression varies across different cell types and is associated with various immune functions, including activation, differentiation, and effector responses, as well as antibody production in plasma cells.

# In[124]:


marker_number = 5
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit variable levels of HLADR expression, particularly in activated or antigen-presenting macrophages involved in immune responses.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with HLADR expression, as it is primarily found on cells involved in immune responses and antigen presentation.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are expected to strongly express HLADR, as it is a characteristic marker of antigen-presenting cells involved in immune surveillance and T cell activation.
# 
# 4. **Tumor**: HLADR expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing HLADR may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with HLADR expression, as HLADR is primarily found on antigen-presenting cells such as dendritic cells, macrophages, and B cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLADR, such as dendritic cells, macrophages, and activated T cells, may exhibit variable levels of HLADR expression, particularly in response to immune activation or inflammatory stimuli.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with HLADR expression, as they are part of the innate immune system and do not play a direct role in antigen presentation.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with HLADR expression, as they are primarily cytotoxic T lymphocytes involved in killing virus-infected or abnormal cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with HLADR expression, as they are primarily involved in immune suppression rather than antigen presentation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with HLADR expression, as they are part of the innate immune system and do not play a direct role in antigen presentation.
# 
# 11. **Plasma**: Plasma cells are not typically associated with HLADR expression, as they are terminally differentiated B cells primarily involved in antibody production.
# 
# 12. **B cell**: B cells may express HLADR, particularly in activated or antigen-presenting subsets involved in immune responses and antibody production.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are expected to strongly express HLADR, as they are specialized antigen-presenting cells involved in antiviral immune responses.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific HLADR expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics would be needed to assess its potential HLADR expression pattern accurately.

# In[125]:


marker_number = 6
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: CD27 expression in macrophages may vary depending on their activation state and tissue microenvironment. While some subsets of activated macrophages may express CD27, its expression is not typically a hallmark feature of macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD27 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express variable levels of CD27, particularly in mature or activated states involved in antigen presentation and immune regulation. However, CD27 expression in dendritic cells is not as common as in lymphocytes.
# 
# 4. **Tumor**: CD27 expression in tumor cells can vary depending on the tumor type and microenvironment. Tumor-infiltrating lymphocytes, particularly activated T cells, may express CD27 and play a role in antitumor immune responses.
# 
# 5. **CD4**: CD4+ T cells may express CD27, especially in memory and activated subsets. CD27 expression on CD4+ T cells is associated with their activation, differentiation, and effector functions.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD27 expression, particularly in activated or memory subsets.
# 
# 7. **NK (Natural Killer) cells**: NK cells may express CD27, particularly in subsets associated with activation and cytotoxicity. CD27 expression on NK cells can modulate their effector functions and interactions with other immune cells.
# 
# 8. **CD8**: CD8+ T cells may express CD27, particularly in memory and activated subsets associated with cytotoxic activity against infected or malignant cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit variable levels of CD27 expression, particularly in activated or memory subsets involved in immune regulation and tolerance.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD27 expression. CD27 is primarily found on lymphoid cells, such as T and B lymphocytes.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD27 expression. CD27 is primarily found on lymphoid cells involved in adaptive immune responses.
# 
# 12. **B cell**: B cells may express CD27, particularly in memory and activated subsets. CD27 expression on B cells is associated with their activation, differentiation into memory cells, and antibody production.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit low to moderate levels of CD27 expression, particularly in response to viral infections or inflammatory stimuli.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD27 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics would be needed to assess its potential CD27 expression pattern accurately.
# 
# Overall, CD27 expression varies across different cell types and is associated with various immune functions, including activation, differentiation, and memory formation in lymphoid cells. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[127]:


marker_number = 7
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# Here's a summary of the potential expression patterns of CD15 in the provided cell types:
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with CD15 expression. CD15 is primarily found on granulocytes, including neutrophils, and is not a characteristic marker of macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD15 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of dendritic cells.
# 
# 4. **Tumor**: CD15 expression in tumor cells can vary depending on the tumor type and differentiation status. In some cases, CD15 expression may be observed in certain types of tumors, particularly those of epithelial origin or with features of squamous differentiation.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of T lymphocytes.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with CD15 expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of NK cells.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of T lymphocytes.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of Tregs.
# 
# 10. **Neutrophil**: Neutrophils are known to express CD15 at high levels, serving as a hallmark marker for their identification and involvement in innate immune responses, particularly in phagocytosis and antimicrobial activity.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of plasma cells.
# 
# 12. **B cell**: B cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of B lymphocytes.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD15 expression. CD15 is primarily found on granulocytes and is not a characteristic marker of pDCs.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD15 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD15 expression accurately. Without such context, it's difficult to provide a definitive answer regarding CD15 expression in BnT cells. If you can provide more details or context about the "BnT" designation, I'd be happy to assist further.

# In[128]:


marker_number = 8
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with CD45RA expression. CD45RA is primarily found on lymphocytes and certain hematopoietic progenitor cells rather than macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD45RA expression. CD45RA is primarily found on cells of the immune system and is not a characteristic marker of mural cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD45RA expression, particularly in immature or precursor subsets. CD45RA expression on dendritic cells may vary depending on their differentiation state and functional specialization.
# 
# 4. **Tumor**: CD45RA expression in tumor cells can vary depending on the tumor type and microenvironment. Tumor-infiltrating immune cells expressing CD45RA may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may exhibit variable levels of CD45RA expression, with naïve CD4+ T cells typically expressing high levels of CD45RA, while memory and activated subsets may downregulate CD45RA expression.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with CD45RA expression. CD45RA is primarily found on lymphocytes rather than antigen-presenting cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD45RA expression. CD45RA is primarily found on T cells and B cells, whereas NK cells have distinct marker expression patterns.
# 
# 8. **CD8**: CD8+ T cells may exhibit variable levels of CD45RA expression, with naïve CD8+ T cells typically expressing high levels of CD45RA, while memory and effector subsets may downregulate CD45RA expression upon activation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit variable levels of CD45RA expression, with some subsets expressing CD45RA as a marker of naïve or resting Tregs, while others may lack CD45RA expression or express other markers associated with activated or suppressive functions.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD45RA expression. CD45RA is primarily found on lymphocytes and certain hematopoietic progenitor cells rather than granulocytes like neutrophils.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD45RA expression. CD45RA is primarily found on lymphocytes and certain hematopoietic progenitor cells rather than terminally differentiated plasma cells.
# 
# 12. **B cell**: B cells may exhibit variable levels of CD45RA expression, with naïve B cells typically expressing high levels of CD45RA, while memory B cells and plasma cells may downregulate CD45RA expression upon activation.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD45RA expression. CD45RA is primarily found on lymphocytes rather than dendritic cell subsets.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD45RA expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD45RA expression pattern accurately.

# In[129]:


marker_number = 9
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: High expression of CD163 is expected in MacCD163 cells, as it serves as a specific marker for a subset of macrophages known for their involvement in tissue repair, inflammation resolution, and scavenging functions.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD163 expression. CD163 is primarily expressed on macrophages and monocytes.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not expected to express CD163, as it is predominantly expressed on cells of the monocyte/macrophage lineage rather than dendritic cells.
# 
# 4. **Tumor**: CD163 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cancers, tumor-associated macrophages (TAMs) may express CD163, contributing to tumor progression and immunosuppression.
# 
# 5. **CD4**: CD4+ T cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage rather than lymphocytes.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not expected to express CD163, as it is primarily associated with monocytes and macrophages.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 8. **CD8**: CD8+ T cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 10. **Neutrophil**: Neutrophils are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 11. **Plasma**: Plasma cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 12. **B cell**: B cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not expected to express CD163. CD163 is primarily associated with cells of the monocyte/macrophage lineage.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD163 expression patterns without further context or clarification. If "BnT" refers to a specific cell type, additional information would be needed to assess its potential CD163 expression pattern accurately.
# 
# Overall, CD163 expression is primarily associated with cells of the monocyte/macrophage lineage, particularly macrophages, and is not typically expressed by other immune cell types such as dendritic cells, lymphocytes, or neutrophils.

# In[130]:


marker_number = 10
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit moderate to high expression of B2M, as B2M is involved in the assembly and expression of major histocompatibility complex class I (MHC-I) molecules, which are crucial for antigen presentation by macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation by immune cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express moderate to high levels of B2M, as B2M is essential for the stability and expression of MHC-I molecules, which play a critical role in dendritic cell-mediated antigen presentation to T cells.
# 
# 4. **Tumor**: B2M expression in tumor cells can vary depending on the tumor type and microenvironment. B2M is necessary for MHC-I expression, and its downregulation or loss in tumor cells can impair immune recognition and facilitate immune evasion.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation, which is crucial for CD8+ T cell activation and cytotoxicity.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit moderate to high levels of B2M, as B2M is essential for the stability and expression of MHC-II molecules, which are involved in antigen presentation to CD4+ T cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation, which is crucial for NK cell recognition of target cells.
# 
# 8. **CD8**: CD8+ T cells may express moderate to high levels of B2M, as B2M is essential for the stability and expression of MHC-I molecules, which present antigens to CD8+ T cells for recognition and activation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation, which is crucial for effector T cell activation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation by professional antigen-presenting cells.
# 
# 11. **Plasma**: Plasma cells are not typically associated with B2M expression. B2M is primarily involved in MHC-I antigen presentation, which is crucial for T cell activation and immune responses.
# 
# 12. **B cell**: B cells may express moderate to high levels of B2M, as B2M is involved in MHC-I antigen presentation, which is important for B cell interactions with CD8+ T cells and immune surveillance.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may express moderate to high levels of B2M, as B2M is essential for the stability and expression of MHC-II molecules, which are involved in antigen presentation to CD4+ T cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific B2M expression patterns without further context or clarification.
# 
# Overall, B2M expression is primarily associated with cells involved in antigen presentation, particularly MHC-I expression, and is crucial for immune recognition and surveillance.

# In[131]:


marker_number = 11
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not expected to express CD20, as CD20 is primarily associated with B cells and certain lymphoid malignancies.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD20 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not expected to express CD20, as it is primarily found on B cells and is involved in B cell activation and function.
# 
# 4. **Tumor**: CD20 expression in tumor cells can vary widely depending on the tumor type and origin. Some B cell lymphomas may express CD20, while other tumor types are unlikely to exhibit CD20 expression.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD20 expression, as CD20 is primarily found on B cells and is involved in B cell receptor signaling and activation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not expected to express CD20.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD20 expression, as it is primarily found on B cells and is involved in B cell development and function.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD20 expression, as it is primarily found on B cells and is involved in B cell receptor signaling and activation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not expected to express CD20, as it is primarily found on B cells and is involved in B cell activation and function.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD20 expression.
# 
# 11. **Plasma**: Plasma cells may exhibit low levels of CD20 expression, particularly in early stages of differentiation, but expression is usually downregulated as plasma cells mature and become antibody-secreting.
# 
# 12. **B cell**: B cells are expected to strongly express CD20, as it is a hallmark marker of B cells and is involved in B cell receptor signaling and activation.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not expected to express CD20.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD20 expression patterns without further context or clarification.
# 
# In summary, CD20 expression is primarily associated with B cells and certain B cell malignancies, while other cell types are not expected to express CD20 under normal physiological conditions.

# In[132]:


marker_number = 12
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: High expression of CD68 is expected in macrophages, including those expressing CD163. CD68 is a well-established marker for macrophages and is involved in various functions such as phagocytosis and antigen presentation.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit low to moderate levels of CD68 expression, particularly in subsets involved in antigen presentation and immune regulation. However, CD68 is more commonly associated with macrophages than dendritic cells.
# 
# 4. **Tumor**: CD68 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-associated macrophages expressing CD68 may infiltrate the tumor mass and contribute to tumor progression or suppression.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit low levels of CD68 expression, particularly in subsets involved in antigen presentation and immune activation.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 12. **B cell**: B cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD68 expression. CD68 is primarily found on cells of the monocyte/macrophage lineage.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD68 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD68 expression pattern accurately. Without such context, it's difficult to provide a definitive answer regarding CD68 expression in BnT cells.
# 
# Overall, CD68 expression is primarily associated with cells of the monocyte/macrophage lineage, including macrophages and certain dendritic cell subsets. Its expression in other cell types, such as tumor cells or lymphocytes, is less common and may vary depending on specific contexts or pathological conditions.

# In[133]:


marker_number = 13
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages, especially those expressing CD163, may exhibit variable levels of IDO1 expression, particularly in response to immunomodulatory signals or in inflammatory environments. IDO1 in macrophages can contribute to immune regulation and tolerance induction.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with IDO1 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express IDO1, particularly in subsets involved in immune regulation and tolerance induction. IDO1 expression in dendritic cells contributes to the suppression of T cell responses and maintenance of immune homeostasis.
# 
# 4. **Tumor**: Tumor cells and tumor-infiltrating immune cells may exhibit variable levels of IDO1 expression. IDO1 expression in tumors is associated with immune evasion mechanisms and suppression of antitumor immune responses.
# 
# 5. **CD4**: CD4+ T cells may express IDO1, particularly in subsets associated with regulatory or suppressive functions. IDO1 expression in CD4+ T cells contributes to the modulation of immune responses and tolerance induction.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of IDO1 expression, particularly in the context of immune regulation and tolerance induction.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with IDO1 expression.
# 
# 8. **CD8**: CD8+ T cells may express IDO1, particularly in subsets associated with regulatory or suppressive functions. IDO1 expression in CD8+ T cells contributes to immune regulation and tolerance induction.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are known to express IDO1, contributing to their immunosuppressive functions and maintenance of immune tolerance.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with IDO1 expression.
# 
# 11. **Plasma**: Plasma cells are not typically associated with IDO1 expression.
# 
# 12. **B cell**: B cells are not typically associated with IDO1 expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may express IDO1, particularly in response to immunomodulatory signals or in the context of immune regulation.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific IDO1 expression patterns without further context or clarification.
# 
# Overall, IDO1 expression varies across different cell types and is associated with immune regulation, tolerance induction, and immune evasion mechanisms, particularly in the context of tumors and regulatory immune cell subsets.

# In[134]:


marker_number = 14
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in T cell receptor (TCR) signaling.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD3 expression. CD3 is primarily found on lymphocytes and is involved in TCR signaling.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 4. **Tumor**: CD3 expression in tumors can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating T cells expressing CD3 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells express CD3 as part of the TCR complex, which plays a crucial role in antigen recognition and T cell activation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 8. **CD8**: CD8+ T cells express CD3 as part of the TCR complex, which plays a crucial role in antigen recognition and T cell activation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells express CD3 as part of the TCR complex, which is involved in immune regulation and tolerance induction.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 12. **B cell**: B cells are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD3 expression. CD3 is primarily found on T cells and is involved in TCR signaling.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD3 expression patterns without further context or clarification.
# 
# Overall, CD3 expression is primarily associated with T cells and is involved in antigen recognition and T cell activation through the TCR complex. It is not typically expressed on other immune cell types such as macrophages, dendritic cells, NK cells, neutrophils, plasma cells, or B cells.

# In[135]:


marker_number = 15
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit low to negligible expression of LAG3. LAG3 is primarily associated with lymphocytes and is not typically expressed by macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with LAG3 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit low to moderate levels of LAG3 expression, particularly in mature or activated subsets involved in antigen presentation and immune regulation.
# 
# 4. **Tumor**: LAG3 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating lymphocytes expressing LAG3 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells, especially regulatory T cells (Tregs), may exhibit high levels of LAG3 expression. LAG3 is a key marker of T cell exhaustion and regulatory T cell function.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit low to moderate levels of LAG3 expression, particularly in activated or inflammatory states.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with LAG3 expression.
# 
# 8. **CD8**: CD8+ T cells, particularly exhausted or dysfunctional subsets, may exhibit high levels of LAG3 expression. LAG3 is often co-expressed with other immune checkpoint molecules on exhausted CD8+ T cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are likely to exhibit high levels of LAG3 expression. LAG3 is a key marker of Treg suppressive function and immune tolerance.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with LAG3 expression.
# 
# 11. **Plasma**: Plasma cells are not typically associated with LAG3 expression.
# 
# 12. **B cell**: B cells are not typically associated with LAG3 expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with LAG3 expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific LAG3 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics would be needed to assess its potential LAG3 expression pattern accurately.
# 
# Overall, LAG3 expression varies across different cell types and is primarily associated with lymphocytes, particularly CD4+ T cells and regulatory T cells, where it plays a crucial role in immune regulation and tolerance.

# In[136]:


marker_number = 16
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages may exhibit variable levels of CD11 expression, particularly in activated states or in response to inflammatory stimuli. CD11 is involved in macrophage adhesion, migration, and phagocytosis.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD11 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express variable levels of CD11, particularly in mature or activated states involved in antigen presentation and immune regulation. CD11 can contribute to dendritic cell migration and interactions with other immune cells.
# 
# 4. **Tumor**: CD11 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD11 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may express low levels of CD11, particularly in subsets associated with tissue homing or migration to inflammatory sites.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD11 expression, particularly under inflammatory or immune stimulatory conditions.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD11 expression.
# 
# 8. **CD8**: CD8+ T cells may express low levels of CD11, particularly in subsets associated with tissue homing or migration to inflammatory sites.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD11 expression.
# 
# 10. **Neutrophil**: Neutrophils are known to express high levels of CD11, particularly CD11b (also known as Mac-1 or CR3), which plays a crucial role in neutrophil adhesion, migration, and phagocytosis.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD11 expression.
# 
# 12. **B cell**: B cells are not typically associated with CD11 expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD11 expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD11 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD11 expression pattern accurately.
# 
# Overall, CD11 expression varies across different cell types and is primarily associated with leukocytes, particularly myeloid cells such as neutrophils and some dendritic cell subsets, where it plays a role in adhesion, migration, and phagocytosis.

# In[137]:


marker_number = 17
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with PD1 expression. PD1 is more commonly found on lymphocytes and is involved in regulating T cell responses.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express PD1, as it is primarily associated with immune cells, particularly T cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit low to moderate levels of PD1 expression, particularly in certain subsets or under specific activation conditions. PD1 expression on dendritic cells could modulate their interactions with T cells and influence immune tolerance.
# 
# 4. **Tumor**: PD1 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor cells or tumor-infiltrating immune cells may express PD1, contributing to immune evasion and resistance to immunotherapy.
# 
# 5. **CD4**: CD4+ T cells, particularly activated or exhausted subsets, may exhibit high levels of PD1 expression. PD1 is involved in regulating T cell activation, tolerance, and exhaustion.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of PD1 expression, particularly in activated or dysfunctional T cell subsets.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with PD1 expression. PD1 is primarily found on T cells and is involved in regulating their function.
# 
# 8. **CD8**: CD8+ T cells, particularly activated or exhausted subsets, may exhibit high levels of PD1 expression. PD1 expression on CD8+ T cells is associated with T cell exhaustion and impaired effector function.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit low levels of PD1 expression, particularly in activated or memory subsets. PD1 expression on Tregs could modulate their suppressive function and immune regulation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with PD1 expression. PD1 is primarily found on lymphocytes and is involved in regulating T cell responses.
# 
# 11. **Plasma**: Plasma cells are not typically associated with PD1 expression. PD1 is primarily found on lymphocytes and is involved in regulating T cell responses.
# 
# 12. **B cell**: B cells are not typically associated with PD1 expression. PD1 is primarily found on lymphocytes and is involved in regulating T cell responses.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with PD1 expression. PD1 is primarily found on lymphocytes and is involved in regulating T cell responses.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific PD1 expression patterns without further context or clarification.
# 
# Overall, PD1 expression is primarily associated with lymphocytes, particularly T cells, and is involved in regulating T cell activation, tolerance, and exhaustion. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[138]:


marker_number = 18
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Low to moderate expression of PDGFRb may be observed in macrophages, particularly in certain subpopulations involved in tissue repair and remodeling.
# 
# 2. **Mural (Mural cells)**: Mural cells, including pericytes and smooth muscle cells, typically exhibit high expression of PDGFRb, as it is a characteristic marker of these cells involved in vascular development and maintenance.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with PDGFRb expression, as it is primarily found on mesenchymal cells rather than cells of hematopoietic origin.
# 
# 4. **Tumor**: PDGFRb expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating stromal cells expressing PDGFRb may influence tumor growth and angiogenesis.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with PDGFRb expression.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with PDGFRb expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with PDGFRb expression.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with PDGFRb expression.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with PDGFRb expression.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with PDGFRb expression.
# 
# 11. **Plasma**: Plasma cells are not typically associated with PDGFRb expression.
# 
# 12. **B cell**: B cells are not typically associated with PDGFRb expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with PDGFRb expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific PDGFRb expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics would be needed to assess its potential PDGFRb expression pattern accurately.
# 
# In summary, PDGFRb expression is primarily associated with mural cells, including pericytes and smooth muscle cells, which play crucial roles in vascular development and maintenance. Its expression in other cell types, particularly immune cells, is not commonly observed.

# In[139]:


marker_number = 19
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with CD7 expression. CD7 is primarily found on lymphoid cells and is not a characteristic marker of macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express CD7. CD7 is primarily found on cells of lymphoid lineage and is not typically expressed by mesenchymal cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD7 expression, particularly in certain subsets or under specific activation conditions. CD7 expression on dendritic cells could modulate their interactions with other immune cells and their migratory properties.
# 
# 4. **Tumor**: CD7 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD7 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells typically express CD7, as it is a characteristic marker of T lymphocytes. CD7 expression on CD4+ T cells is involved in T cell activation, adhesion, and signaling.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD7 expression. CD7 expression on HLA-DR-expressing cells could modulate their immune functions and interactions with other immune cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells may express CD7, particularly in subsets associated with cytotoxic activity and immune surveillance. CD7 expression on NK cells is involved in their effector functions and interactions with target cells.
# 
# 8. **CD8**: CD8+ T cells typically express CD7, similar to CD4+ T cells. CD7 expression on CD8+ T cells is involved in T cell activation, adhesion, and cytotoxicity against infected or malignant cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit variable levels of CD7 expression. CD7 expression on Tregs could modulate their suppressive functions and interactions with other immune cells.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD7 expression. CD7 is primarily found on lymphoid cells and is not expressed by granulocytes such as neutrophils.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD7 expression. CD7 is primarily found on cells of lymphoid lineage and is not expressed by terminally differentiated plasma cells.
# 
# 12. **B cell**: B cells may express CD7, particularly in immature or precursor subsets. CD7 expression on B cells is involved in B cell development, maturation, and signaling.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit variable levels of CD7 expression. CD7 expression on pDCs could modulate their immune functions and interactions with other immune cells.
# 
# 14. **BnT**: Without further context or clarification regarding the "BnT" designation, it's challenging to infer specific CD7 expression patterns. If "BnT" refers to a specific cell type or population, additional information would be needed to assess its potential CD7 expression accurately.
# 
# Overall, CD7 expression varies across different cell types and is associated with various immune functions, including T cell activation, adhesion, and signaling. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[140]:


marker_number = 20
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# ##### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit low to moderate levels of GrzB expression, particularly in activated or inflammatory macrophages involved in immune responses and cytotoxic activity.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with GrzB expression, as GrzB is primarily found in immune cells involved in cytotoxicity and immune surveillance.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with GrzB expression, as their primary function revolves around antigen presentation and immune activation rather than cytotoxicity.
# 
# 4. **Tumor**: GrzB expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing GrzB, such as cytotoxic lymphocytes, may play a role in antitumor immunity.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with GrzB expression, as their effector functions primarily involve cytokine secretion rather than cytotoxicity.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with GrzB expression, as HLA-DR is primarily involved in antigen presentation rather than cytotoxic activity.
# 
# 7. **NK (Natural Killer) cells**: NK cells are known to express GrzB, which is essential for their cytotoxic activity against virus-infected or transformed cells.
# 
# 8. **CD8**: CD8+ T cells are known to express high levels of GrzB, which is critical for their cytotoxic function against infected or malignant cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with GrzB expression, as their primary function is immunosuppressive rather than cytotoxic.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with GrzB expression, as GrzB is primarily found in lymphocytes, particularly cytotoxic T cells and NK cells.
# 
# 11. **Plasma**: Plasma cells are not typically associated with GrzB expression, as their primary function is antibody production rather than cytotoxicity.
# 
# 12. **B cell**: B cells are not typically associated with GrzB expression, as their primary function is antibody production rather than cytotoxicity.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with GrzB expression, as their primary function is type I interferon production rather than cytotoxic activity.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific GrzB expression patterns without further context or clarification.
# 
# Overall, Granzyme B expression is primarily associated with cytotoxic immune cells such as NK cells and cytotoxic T lymphocytes (CTLs), which play a crucial role in immune surveillance and elimination of infected or malignant cells.

# In[141]:


marker_number = 21
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit variable levels of PD-L1 expression, particularly in response to inflammatory stimuli or immunoregulatory signals. PD-L1 expression on macrophages can modulate T cell activation and immune responses.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with PD-L1 expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of PD-L1 expression, particularly in mature or activated states. PD-L1 expression on dendritic cells is involved in immune tolerance, T cell regulation, and immune evasion mechanisms.
# 
# 4. **Tumor**: PD-L1 expression in tumor cells can vary widely depending on the tumor type, stage, and microenvironment. PD-L1 expression on tumor cells can suppress antitumor immune responses by engaging with PD-1 receptors on T cells, leading to T cell exhaustion and immune evasion.
# 
# 5. **CD4**: CD4+ T cells may express PD-L1 under certain conditions, particularly in regulatory T cell subsets or upon activation by immune signaling pathways. PD-L1 expression on CD4+ T cells can modulate immune responses and T cell regulation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of PD-L1 expression, particularly in response to inflammatory or immunoregulatory signals.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with PD-L1 expression. PD-L1 expression is more commonly found on antigen-presenting cells and tumor cells.
# 
# 8. **CD8**: CD8+ T cells may exhibit PD-L1 expression, particularly in exhausted or dysfunctional subsets within the tumor microenvironment. PD-L1 expression on CD8+ T cells can impair their effector functions and antitumor responses.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may express PD-L1 as part of their immunosuppressive mechanisms to inhibit effector T cell responses and maintain immune tolerance.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with PD-L1 expression. PD-L1 expression is more commonly found on cells involved in antigen presentation and immune regulation.
# 
# 11. **Plasma**: Plasma cells are not typically associated with PD-L1 expression. PD-L1 expression is more commonly found on cells involved in immune regulation and immune evasion.
# 
# 12. **B cell**: B cells are not typically associated with PD-L1 expression. PD-L1 expression is more commonly found on antigen-presenting cells and tumor cells.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit variable levels of PD-L1 expression, particularly in response to viral infections or inflammatory stimuli. PD-L1 expression on pDCs can modulate immune responses and T cell regulation.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific PD-L1 expression patterns without further context or clarification.
# 
# Overall, PD-L1 expression varies across different cell types and is associated with various immune functions, including immune regulation, T cell activation, and immune evasion in the context of cancer. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[142]:


marker_number = 22
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# 1. **MacCD163 (Macrophage CD163)**: Macrophages are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express TCF7, as it is primarily associated with immune cell function.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 4. **Tumor**: TCF7 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor cells may exhibit altered expression of TCF7, which could influence tumor growth and progression.
# 
# 5. **CD4**: CD4+ T cells are likely to express TCF7, as it plays a crucial role in T cell development and differentiation, particularly in the generation and maintenance of memory T cell populations.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with TCF7 expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 8. **CD8**: CD8+ T cells are likely to express TCF7, as it is involved in T cell development and differentiation, particularly in the generation and maintenance of memory CD8+ T cell populations.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit variable levels of TCF7 expression, as it is involved in T cell development and differentiation, but its role in Treg biology is less well-characterized compared to conventional T cells.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 11. **Plasma**: Plasma cells are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 12. **B cell**: B cells are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with TCF7 expression, as it is primarily found in T cells and involved in their development and differentiation.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific TCF7 expression patterns without further context or clarification.
# 
# Overall, TCF7 expression is primarily associated with T cell development and differentiation, particularly in CD4+ and CD8+ T cells, where it plays crucial roles in generating and maintaining memory T cell populations. Its expression in other cell types is less common and may indicate context-specific roles or aberrant expression patterns.

# In[143]:


marker_number = 23
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages may exhibit low to moderate expression of CD45RO, particularly in activated or inflammatory states. CD45RO expression in macrophages may indicate their involvement in immune responses and antigen presentation.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD45RO expression.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD45RO expression, particularly in mature or activated subsets involved in antigen presentation and immune regulation.
# 
# 4. **Tumor**: CD45RO expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD45RO may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may exhibit moderate to high expression of CD45RO, particularly in memory and activated subsets involved in adaptive immune responses.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD45RO expression, particularly in memory and activated subsets.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD45RO expression.
# 
# 8. **CD8**: CD8+ T cells may exhibit low to moderate expression of CD45RO, particularly in memory and activated subsets involved in cytotoxic activity against infected or malignant cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit variable levels of CD45RO expression, particularly in activated or memory subsets associated with immune regulation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD45RO expression.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD45RO expression.
# 
# 12. **B cell**: B cells, particularly memory B cell subsets, may exhibit moderate to high expression of CD45RO, reflecting their involvement in adaptive immune responses and antibody production.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD45RO expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD45RO expression patterns without further context or clarification.
# 
# Overall, CD45RO expression varies across different cell types and is often associated with memory T cells and activated immune cells involved in adaptive immune responses. Its expression pattern can provide insights into the immune status and functionality of the analyzed cell populations in different physiological and pathological contexts.

# In[144]:


marker_number = 24
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# ##### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not expected to express FOXP3, as FOXP3 is primarily associated with regulatory T cells (Tregs) and is a key transcription factor involved in their development and function.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with FOXP3 expression, as FOXP3 is primarily found in immune cells, particularly Tregs.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not expected to express FOXP3 under normal physiological conditions. FOXP3 expression is mainly associated with regulatory T cells and is involved in maintaining immune tolerance and suppressing excessive immune responses.
# 
# 4. **Tumor**: FOXP3 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating regulatory T cells expressing FOXP3 may contribute to immune suppression and tumor immune evasion.
# 
# 5. **CD4**: FOXP3 is predominantly expressed in CD4+ regulatory T cells (Tregs). Therefore, CD4+ T cells expressing FOXP3 are likely to represent regulatory T cells involved in immune regulation and tolerance.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not expected to express FOXP3. FOXP3 expression is specific to regulatory T cells and is not typically found in antigen-presenting cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express FOXP3, as FOXP3 is primarily associated with T cell subsets, particularly regulatory T cells.
# 
# 8. **CD8**: FOXP3 expression is not typically associated with CD8+ T cells. CD8+ T cells are primarily cytotoxic T lymphocytes involved in cell-mediated immune responses and are not known to express FOXP3.
# 
# 9. **Treg (Regulatory T cell)**: FOXP3 is a key marker of regulatory T cells, and high expression of FOXP3 is expected in this cell population. Tregs play a crucial role in immune regulation and maintaining self-tolerance.
# 
# 10. **Neutrophil**: Neutrophils are not expected to express FOXP3, as FOXP3 is primarily associated with lymphoid lineage cells, particularly regulatory T cells.
# 
# 11. **Plasma**: Plasma cells are not expected to express FOXP3, as FOXP3 is primarily associated with regulatory T cells involved in immune regulation and tolerance.
# 
# 12. **B cell**: B cells are not expected to express FOXP3, as FOXP3 is primarily associated with T cell subsets, particularly regulatory T cells.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not expected to express FOXP3, as FOXP3 is primarily associated with regulatory T cells and is involved in immune regulation rather than antigen presentation.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific FOXP3 expression patterns without further context or clarification.
# 
# In summary, FOXP3 expression is primarily associated with regulatory T cells (Tregs) and is involved in immune regulation and tolerance. While other cell types are not typically associated with FOXP3 expression, its presence in certain tumors or under specific pathological conditions may indicate immune regulatory mechanisms within the tumor microenvironment.

# In[145]:


marker_number = 25
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages are not typically associated with ICOS expression, as ICOS is primarily found on activated T cells and some B cell subsets.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express ICOS, as it is primarily a T cell co-stimulatory molecule.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with ICOS expression, as ICOS is primarily found on activated T cells and plays a role in T cell activation and differentiation.
# 
# 4. **Tumor**: ICOS expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing ICOS may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may exhibit ICOS expression, particularly in activated or memory subsets. ICOS expression on CD4+ T cells is associated with their effector functions and regulation of immune responses.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit ICOS expression, particularly in activated T cell subsets involved in immune responses.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with ICOS expression, as it is primarily found on activated T cells.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with ICOS expression, as ICOS is primarily found on CD4+ T cells and plays a role in CD4+ T cell activation and differentiation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit ICOS expression, particularly in subsets associated with suppressive functions and immune regulation.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with ICOS expression, as it is primarily found on lymphoid cells involved in adaptive immune responses.
# 
# 11. **Plasma**: Plasma cells are not typically associated with ICOS expression, as it is primarily found on T cells.
# 
# 12. **B cell**: B cells may exhibit ICOS expression, particularly in activated or memory B cell subsets. ICOS expression on B cells is associated with their activation and differentiation into antibody-producing plasma cells.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with ICOS expression, as it is primarily found on activated T cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific ICOS expression patterns without further context or clarification.
# 
# Overall, ICOS expression is primarily associated with activated T cells and some B cell subsets, playing a role in T cell activation, differentiation, and regulation of immune responses. Its expression in other cell types is less common and may vary depending on specific contexts and conditions.

# In[148]:


marker_number = 26
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not expected to express CD8a, as CD8a is primarily associated with T lymphocytes and not macrophages.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD8a expression. CD8a is primarily found on T cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not expected to express CD8a, as it is primarily associated with T lymphocytes.
# 
# 4. **Tumor**: CD8a expression in tumors can vary depending on the presence of tumor-infiltrating lymphocytes (TILs), particularly CD8+ cytotoxic T cells. High levels of CD8a expression in tumors are associated with a favorable prognosis and enhanced response to immunotherapy.
# 
# 5. **CD4**: CD4+ T cells are not expected to express CD8a. CD8a is primarily associated with CD8+ cytotoxic T cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not expected to express CD8a.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express CD8a. CD8a is primarily associated with T lymphocytes.
# 
# 8. **CD8**: CD8+ cytotoxic T cells are expected to strongly express CD8a. CD8a is a characteristic marker of cytotoxic T cells involved in cell-mediated immunity.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not expected to express CD8a. CD8a is primarily associated with CD8+ cytotoxic T cells.
# 
# 10. **Neutrophil**: Neutrophils are not expected to express CD8a. CD8a is primarily associated with T lymphocytes.
# 
# 11. **Plasma**: Plasma cells are not expected to express CD8a. CD8a is primarily associated with T lymphocytes.
# 
# 12. **B cell**: B cells are not expected to express CD8a. CD8a is primarily associated with T lymphocytes.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not expected to express CD8a. CD8a is primarily associated with T lymphocytes.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD8a expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD8a expression pattern accurately.
# 
# Overall, CD8a expression is primarily associated with CD8+ cytotoxic T cells and is not expected to be expressed in other cell types listed.

# In[149]:


marker_number = 27
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages may express carbonic anhydrase, particularly in specialized subpopulations involved in specific physiological functions such as pH regulation within tissue microenvironments or bone resorption.
# 
# 2. **Mural (Mural cells)**: Mural cells, including pericytes and smooth muscle cells, may express carbonic anhydrase, contributing to pH regulation and ion transport in vascular and tissue environments.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of carbonic anhydrase expression, potentially playing a role in intracellular pH regulation and dendritic cell function.
# 
# 4. **Tumor**: Carbonic anhydrase expression in tumor cells can vary depending on the tumor type and microenvironment. Some tumors may upregulate carbonic anhydrase expression to facilitate pH regulation and tumor cell survival in hypoxic conditions.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with carbonic anhydrase expression.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of carbonic anhydrase expression, potentially contributing to intracellular pH regulation and immune cell function.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with carbonic anhydrase expression.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with carbonic anhydrase expression.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with carbonic anhydrase expression.
# 
# 10. **Neutrophil**: Neutrophils may express carbonic anhydrase, potentially contributing to intracellular pH regulation and the respiratory burst reaction during the immune response.
# 
# 11. **Plasma**: Plasma cells are not typically associated with carbonic anhydrase expression.
# 
# 12. **B cell**: B cells are not typically associated with carbonic anhydrase expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with carbonic anhydrase expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific carbonic anhydrase expression patterns without further context or clarification.
# 
# Overall, carbonic anhydrase expression may be observed in certain immune cells and tissue-resident cells, contributing to pH regulation and cellular homeostasis in physiological and pathological conditions. However, its expression pattern can vary widely depending on cell type, tissue context, and environmental factors.

# In[150]:


marker_number = 28
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit moderate to high levels of CD33 expression, as CD33 is commonly found on myeloid cells, including macrophages, and is involved in cell adhesion, phagocytosis, and immune regulation.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD33 expression. CD33 is primarily found on myeloid cells rather than cells of mesenchymal origin.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express variable levels of CD33, particularly in subsets involved in antigen presentation and immune regulation. CD33 expression on dendritic cells could modulate their interactions with other immune cells and their ability to initiate immune responses.
# 
# 4. **Tumor**: CD33 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD33 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than adaptive immune cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD33 expression, particularly in activated or inflammatory states.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than lymphoid cells.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than T lymphocytes.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than regulatory T cells.
# 
# 10. **Neutrophil**: Neutrophils are known to express CD33, as it is a marker commonly found on myeloid cells, including granulocytes. CD33 may play a role in neutrophil adhesion and activation.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than plasma cells.
# 
# 12. **B cell**: B cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than B lymphocytes.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD33 expression. CD33 is more commonly found on myeloid cells involved in innate immune responses rather than plasmacytoid dendritic cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD33 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential CD33 expression pattern accurately. Without such context, it's difficult to provide a definitive answer regarding CD33 expression in BnT cells. If you can provide more details or context about the "BnT" designation, I'd be happy to assist further.

# In[151]:


marker_number = 29
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not typically associated with high levels of Ki67 expression, as they are often involved in tissue maintenance, phagocytosis, and immune regulation rather than active proliferation.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express Ki67 at significant levels, as they are primarily involved in structural support and tissue stability rather than active proliferation.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of Ki67 expression, particularly in subsets undergoing maturation or activation in response to inflammatory stimuli or antigen encounter.
# 
# 4. **Tumor**: Ki67 expression in tumor cells can vary widely depending on the tumor type and aggressiveness. High Ki67 expression is often associated with increased proliferation and poor prognosis in various cancers.
# 
# 5. **CD4**: Ki67 expression may be observed in subsets of activated CD4+ T cells, particularly in effector or memory T cell populations responding to antigenic stimulation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of Ki67 expression, particularly in response to immune activation and antigen presentation.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with high levels of Ki67 expression, as they are primarily involved in cytotoxicity and immune surveillance rather than active proliferation.
# 
# 8. **CD8**: Ki67 expression may be observed in subsets of activated or memory CD8+ T cells, particularly during immune responses against pathogens or tumors.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with high levels of Ki67 expression, as they are primarily involved in immune suppression and maintenance of peripheral tolerance rather than proliferation.
# 
# 10. **Neutrophil**: Neutrophils are short-lived cells and do not typically express Ki67 at significant levels, as they primarily undergo differentiation and activation in the bone marrow before being released into the circulation.
# 
# 11. **Plasma**: Plasma cells may exhibit low levels of Ki67 expression, particularly in subsets undergoing differentiation from activated B cells. However, Ki67 expression is generally low in terminally differentiated plasma cells dedicated to antibody production.
# 
# 12. **B cell**: Ki67 expression may be observed in subsets of activated B cells, particularly during germinal center reactions and antibody affinity maturation processes.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit low levels of Ki67 expression, particularly in response to viral infections or inflammatory stimuli.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific Ki67 expression patterns without further context or clarification.
# 
# Overall, Ki67 expression varies across different cell types and is associated with active proliferation and cell cycle progression. Its expression pattern can provide insights into the proliferative status and activation state of the analyzed cell populations in different physiological and pathological contexts.

# In[152]:


marker_number = 30
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit variable levels of VISTA expression, particularly in response to immune activation or inflammation. VISTA expression in macrophages can modulate their interactions with T cells and influence immune responses.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with VISTA expression. VISTA is primarily found on immune cells and plays a role in immune regulation rather than mesenchymal cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may express VISTA, particularly in certain subsets or under specific activation conditions. VISTA expression on dendritic cells can modulate their antigen-presenting functions and T cell priming.
# 
# 4. **Tumor**: VISTA expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor cells may upregulate VISTA as a mechanism to evade immune surveillance and suppress antitumor immune responses.
# 
# 5. **CD4**: CD4+ T cells may exhibit VISTA expression, particularly in regulatory T cell (Treg) subsets. VISTA expression on Tregs is involved in their suppressive functions and regulation of immune tolerance.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit VISTA expression, particularly in the context of immune activation and regulation.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with VISTA expression. VISTA is primarily found on T cells and myeloid cells involved in immune regulation.
# 
# 8. **CD8**: CD8+ T cells may exhibit variable levels of VISTA expression, particularly in subsets associated with immune regulation and exhaustion. VISTA expression on CD8+ T cells can modulate their cytotoxic functions and responsiveness to immune checkpoints.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are known to express VISTA, which contributes to their suppressive functions and maintenance of immune tolerance. VISTA expression on Tregs can inhibit effector T cell responses and promote immune homeostasis.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with VISTA expression. VISTA is primarily found on lymphoid and myeloid cells involved in adaptive immune responses.
# 
# 11. **Plasma**: Plasma cells are not typically associated with VISTA expression. VISTA is primarily found on cells involved in immune regulation and tolerance.
# 
# 12. **B cell**: B cells are not typically associated with VISTA expression. VISTA is primarily found on T cells and myeloid cells involved in immune regulation and tolerance.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with VISTA expression. VISTA is primarily found on conventional dendritic cells and myeloid cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific VISTA expression patterns without further context or clarification.
# 
# Overall, VISTA expression varies across different cell types and plays a role in immune regulation, tolerance, and evasion of antitumor immune responses. Its expression pattern can provide insights into the immunological characteristics and functions of the analyzed cell populations in different physiological and pathological contexts.

# In[153]:


marker_number = 31
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 may exhibit variable levels of CD40 expression, particularly in response to inflammatory stimuli or interactions with T cells and other immune cells.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD40 expression. CD40 is primarily found on immune cells involved in antigen presentation and immune activation.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD40 expression, particularly in mature or activated states involved in antigen presentation, T cell activation, and immune regulation.
# 
# 4. **Tumor**: CD40 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD40 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells may express CD40, particularly upon activation or in subsets associated with helper functions and immune regulation.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD40 expression, particularly under inflammatory or immune stimulatory conditions.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD40 expression. CD40 is primarily found on antigen-presenting cells and lymphocytes involved in adaptive immune responses.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD40 expression. CD40 is primarily found on antigen-presenting cells and lymphocytes involved in adaptive immune responses.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD40 expression. CD40 is primarily found on antigen-presenting cells and lymphocytes involved in adaptive immune responses.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD40 expression. CD40 is primarily found on antigen-presenting cells and lymphocytes involved in adaptive immune responses.
# 
# 11. **Plasma**: Plasma cells may exhibit low to moderate levels of CD40 expression. CD40 signaling in plasma cells can regulate antibody production and class switching.
# 
# 12. **B cell**: B cells may exhibit moderate to high levels of CD40 expression. CD40 signaling is crucial for B cell activation, differentiation, and antibody production.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may exhibit variable levels of CD40 expression, particularly in response to viral infections or inflammatory stimuli.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD40 expression patterns without further context or clarification.

# In[154]:


marker_number = 32
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages are not typically associated with CD4 expression. CD4 is primarily expressed on helper T cells and a subset of regulatory T cells.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express CD4. CD4 is primarily found on lymphocytes, particularly helper T cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with CD4 expression. CD4 is primarily found on T lymphocytes and is involved in their activation and function.
# 
# 4. **Tumor**: Tumor cells may or may not express CD4, depending on the tumor type and context. CD4 expression in tumors could be indicative of infiltrating lymphocytes or other immune cells.
# 
# 5. **CD4**: CD4+ T cells are expected to strongly express CD4, as it is a defining marker of this T cell subset involved in immune regulation and coordination of immune responses.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with CD4 expression. CD4 is primarily expressed on helper T cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express CD4. CD4 is primarily expressed on T lymphocytes and is involved in their activation and function.
# 
# 8. **CD8**: CD8+ T cells are not expected to express CD4. CD4 is primarily expressed on helper T cells, while CD8 is expressed on cytotoxic T cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may express CD4, as it is a defining marker of this T cell subset involved in immune regulation and suppression of immune responses.
# 
# 10. **Neutrophil**: Neutrophils are not expected to express CD4. CD4 is primarily expressed on T lymphocytes and is involved in their activation and function.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD4 expression. CD4 is primarily expressed on T lymphocytes and is involved in their activation and function.
# 
# 12. **B cell**: B cells are not typically associated with CD4 expression. CD4 is primarily expressed on T lymphocytes and is involved in their activation and function.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not expected to express CD4. CD4 is primarily expressed on T lymphocytes and is involved in their activation and function.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD4 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information would be needed to assess its potential CD4 expression accurately.

# In[155]:


marker_number = 33
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are likely to express CD14, as CD14 is a characteristic marker of monocytes and macrophages involved in innate immune responses and phagocytosis.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD14 expression. CD14 is primarily found on cells of the myeloid lineage, such as monocytes and macrophages.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD14 expression, particularly in certain subsets or under specific activation conditions. CD14 expression on dendritic cells could modulate their interactions with other immune cells and their ability to phagocytose immune complexes.
# 
# 4. **Tumor**: CD14 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-infiltrating immune cells expressing CD14 may influence the tumor immune response.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than lymphocytes.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated monocytes/macrophages, may exhibit variable levels of CD14 expression, particularly under inflammatory or immune stimulatory conditions.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD14 expression. CD14 is primarily found on cells of the myeloid lineage involved in innate immune responses.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than lymphocytes.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than lymphocytes.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD14 expression. CD14 is primarily found on monocytes and macrophages rather than granulocytes like neutrophils.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than lymphocytes or plasma cells.
# 
# 12. **B cell**: B cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than lymphocytes or plasma cells.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD14 expression. CD14 is more commonly found on myeloid cells involved in innate immunity rather than dendritic cells.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD14 expression patterns without further context or clarification.
# 
# In summary, CD14 expression is primarily associated with cells of the myeloid lineage, particularly monocytes and macrophages, involved in innate immune responses and phagocytosis. Its expression in other cell types, such as dendritic cells or tumor cells, may vary depending on specific conditions and contexts.

# In[156]:


marker_number = 34
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: Macrophages may express low to undetectable levels of E-cadherin, as it is not typically a characteristic marker of macrophages. E-cadherin is primarily associated with epithelial cells and plays a role in cell-cell adhesion.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express significant levels of E-cadherin. E-cadherin is primarily found in epithelial tissues rather than in mural cells.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells are not typically associated with E-cadherin expression. They are part of the immune system and are involved in antigen presentation and immune regulation rather than cell-cell adhesion mediated by E-cadherin.
# 
# 4. **Tumor**: E-cadherin expression in tumor cells can vary depending on the tumor type and stage. Loss of E-cadherin expression is often associated with epithelial-mesenchymal transition (EMT), a process implicated in tumor progression and metastasis.
# 
# 5. **CD4**: CD4+ T cells are not expected to express E-cadherin, as it is primarily associated with epithelial cells and mediates cell-cell adhesion rather than immune functions.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, are not typically associated with E-cadherin expression.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express E-cadherin, as it is primarily associated with epithelial tissues rather than immune cells.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with E-cadherin expression. E-cadherin is primarily found in epithelial tissues and plays a role in cell-cell adhesion.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not expected to express E-cadherin, as it is primarily associated with epithelial tissues rather than immune cells.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with E-cadherin expression. E-cadherin is primarily found in epithelial tissues and plays a role in cell-cell adhesion.
# 
# 11. **Plasma**: Plasma cells are not typically associated with E-cadherin expression. E-cadherin is primarily found in epithelial tissues and plays a role in cell-cell adhesion.
# 
# 12. **B cell**: B cells are not typically associated with E-cadherin expression. E-cadherin is primarily found in epithelial tissues and plays a role in cell-cell adhesion.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with E-cadherin expression. E-cadherin is primarily found in epithelial tissues and plays a role in cell-cell adhesion.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific E-cadherin expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential E-cadherin expression accurately.
# 
# Overall, E-cadherin expression is primarily associated with epithelial tissues, where it mediates cell-cell adhesion and tissue integrity. Its expression in immune cells is limited, and it is not typically expressed by the immune cell types listed.

# In[157]:


marker_number = 35
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# CD303, also known as blood dendritic cell antigen 2 (BDCA-2) or C-type lectin domain family 4 member C (CLEC4C), is primarily expressed on plasmacytoid dendritic cells (pDCs).
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 are not expected to express CD303, as CD303 is primarily associated with dendritic cells, particularly plasmacytoid dendritic cells.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not expected to express CD303, as it is primarily found on immune cells, particularly dendritic cells.
# 
# 3. **DC (Dendritic Cell)**: Plasmacytoid dendritic cells (pDCs) are expected to strongly express CD303. CD303 is a specific marker for pDCs and is involved in their functions, including the production of type I interferons in response to viral infections.
# 
# 4. **Tumor**: CD303 expression in tumor cells is not expected, as it is primarily associated with dendritic cells rather than tumor cells.
# 
# 5. **CD4**: CD4+ T cells are not expected to express CD303. CD303 is primarily found on dendritic cells, including plasmacytoid dendritic cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD303 expression. However, CD303 is more commonly associated with plasmacytoid dendritic cells.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not expected to express CD303. CD303 is primarily associated with dendritic cells rather than innate immune cells like NK cells.
# 
# 8. **CD8**: CD8+ T cells are not expected to express CD303. CD303 is primarily found on dendritic cells, particularly plasmacytoid dendritic cells.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not expected to express CD303. CD303 is primarily associated with dendritic cells rather than T regulatory cells.
# 
# 10. **Neutrophil**: Neutrophils are not expected to express CD303. CD303 is primarily found on dendritic cells and is not associated with neutrophil function.
# 
# 11. **Plasma**: Plasma cells are not expected to express CD303. CD303 is primarily associated with dendritic cells rather than plasma cells.
# 
# 12. **B cell**: B cells are not expected to express CD303. CD303 is primarily found on dendritic cells and is not associated with B cell function.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells (pDCs) are expected to strongly express CD303. CD303 is a specific marker for pDCs and is involved in their functions, including the production of type I interferons in response to viral infections.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific CD303 expression patterns without further context or clarification.
# 
# In summary, CD303 expression is primarily associated with plasmacytoid dendritic cells (pDCs), and it is not typically expressed on other cell types such as macrophages, T cells, B cells, or neutrophils.

# In[158]:


marker_number = 36
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 1. **MacCD163 (Macrophage CD163)**: High expression of CD206 may be observed in macrophages expressing CD163. CD206, also known as mannose receptor, is commonly found on macrophages and is involved in the recognition and phagocytosis of glycoproteins and pathogens.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with CD206 expression. CD206 is primarily found on macrophages and dendritic cells involved in antigen uptake and clearance.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit variable levels of CD206 expression, particularly in certain subsets or under specific activation conditions. CD206 expression on dendritic cells is involved in antigen uptake and presentation to T cells.
# 
# 4. **Tumor**: CD206 expression in tumor cells can vary depending on the tumor type and microenvironment. In some cases, tumor-associated macrophages expressing CD206 may influence tumor progression and immune evasion.
# 
# 5. **CD4**: CD4+ T cells are not typically associated with CD206 expression. CD206 is primarily found on antigen-presenting cells such as macrophages and dendritic cells.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit variable levels of CD206 expression, particularly in activated or inflammatory states.
# 
# 7. **NK (Natural Killer) cells**: NK cells are not typically associated with CD206 expression. CD206 is primarily found on antigen-presenting cells involved in innate and adaptive immune responses.
# 
# 8. **CD8**: CD8+ T cells are not typically associated with CD206 expression. CD206 is primarily found on antigen-presenting cells involved in immune surveillance and antigen presentation.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells are not typically associated with CD206 expression. CD206 is primarily found on antigen-presenting cells involved in immune regulation and tolerance induction.
# 
# 10. **Neutrophil**: Neutrophils are not typically associated with CD206 expression. CD206 is primarily found on macrophages and dendritic cells involved in antigen uptake and immune responses.
# 
# 11. **Plasma**: Plasma cells are not typically associated with CD206 expression. CD206 is primarily found on antigen-presenting cells involved in immune surveillance and antigen presentation.
# 
# 12. **B cell**: B cells are not typically associated with CD206 expression. CD206 is primarily found on macrophages and dendritic cells involved in antigen uptake and immune responses.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells are not typically associated with CD206 expression. CD206 is primarily found on myeloid dendritic cells involved in antigen presentation and immune activation.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation, it's challenging to infer specific CD206 expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics would be needed to assess its potential CD206 expression pattern accurately.

# In[159]:


marker_number = 37
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# Cleaved PARP (Poly ADP-ribose polymerase) is a marker of apoptosis, often used to detect cells undergoing programmed cell death.
# 
# 1. **MacCD163 (Macrophage CD163)**: Macrophages expressing CD163 might show minimal to no expression of cleaved PARP under normal physiological conditions. However, in response to apoptotic stimuli or inflammatory signals, macrophages may undergo apoptosis, leading to cleaved PARP expression.
# 
# 2. **Mural (Mural cells)**: Mural cells, such as pericytes and smooth muscle cells, are not typically associated with cleaved PARP expression as they are not actively undergoing apoptosis under normal circumstances.
# 
# 3. **DC (Dendritic Cell)**: Dendritic cells may exhibit minimal cleaved PARP expression under steady-state conditions. However, upon encountering apoptotic signals or during cellular stress, dendritic cells can undergo apoptosis, leading to cleaved PARP expression.
# 
# 4. **Tumor**: Tumor cells may exhibit variable levels of cleaved PARP expression depending on the tumor type, stage, and microenvironment. Apoptosis evasion is a hallmark of cancer, but certain treatments such as chemotherapy or targeted therapy can induce apoptosis in tumor cells, leading to cleaved PARP expression as a marker of treatment response.
# 
# 5. **CD4**: CD4+ T cells might exhibit minimal cleaved PARP expression under normal physiological conditions. However, in response to apoptotic stimuli or activation-induced cell death, CD4+ T cells undergoing apoptosis may show cleaved PARP expression.
# 
# 6. **HLADR (Human Leukocyte Antigen-DR)**: Cells expressing HLA-DR, such as dendritic cells and activated T cells, may exhibit cleaved PARP expression under conditions of cellular stress, apoptosis induction, or immune activation.
# 
# 7. **NK (Natural Killer) cells**: NK cells may show minimal cleaved PARP expression under normal conditions. However, in response to apoptotic signals or during target cell recognition and killing, NK cells may undergo apoptosis, leading to cleaved PARP expression.
# 
# 8. **CD8**: CD8+ T cells might exhibit minimal cleaved PARP expression under steady-state conditions. However, during activation-induced cell death or in response to apoptotic signals, CD8+ T cells undergoing apoptosis may show cleaved PARP expression.
# 
# 9. **Treg (Regulatory T cell)**: Regulatory T cells may exhibit minimal cleaved PARP expression under normal physiological conditions. However, in response to apoptotic stimuli or during regulation of immune responses, Tregs undergoing apoptosis may show cleaved PARP expression.
# 
# 10. **Neutrophil**: Neutrophils are short-lived cells and may undergo apoptosis following activation or at the end of their lifespan. Therefore, neutrophils may show cleaved PARP expression as they undergo programmed cell death.
# 
# 11. **Plasma**: Plasma cells are long-lived and terminally differentiated cells that typically do not undergo apoptosis under normal physiological conditions. Hence, cleaved PARP expression is not commonly observed in plasma cells.
# 
# 12. **B cell**: B cells may exhibit minimal cleaved PARP expression under normal physiological conditions. However, in response to apoptotic signals or during negative selection in the bone marrow or peripheral lymphoid organs, B cells undergoing apoptosis may show cleaved PARP expression.
# 
# 13. **pDC (Plasmacytoid Dendritic Cell)**: Plasmacytoid dendritic cells may show minimal cleaved PARP expression under steady-state conditions. However, upon encountering apoptotic signals or during cellular stress, pDCs undergoing apoptosis may exhibit cleaved PARP expression.
# 
# 14. **BnT**: As "BnT" is not a standard cell type designation commonly used in immunology or cell biology, it's challenging to infer specific cleaved PARP expression patterns without further context or clarification. If "BnT" refers to a specific cell type or population, additional information about its characteristics, lineage, or function would be needed to assess its potential cleaved PARP expression accurately.
# 
# Overall, the expression of cleaved PARP varies across different cell types and is associated with apoptotic processes, cellular stress, and immune responses. Its expression pattern can provide insights into the physiological state and dynamics of cell populations under different conditions, including tissue homeostasis, immune activation, and disease progression.

# In[160]:


marker_number = 38
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 

# In[162]:


marker_number = 39
make_boxplots_per_celltype(df, 'cell_labels', marker_names[marker_number], f'{marker_names[marker_number]} marker by cell type', display_desc=True)


# #### Interpretation
# 

# ## 2. High level correlation analysis
# REQUIREMENT: Correlation patterns between markers and cell types (at least 3) with a biological explanation,

# ### Independent obs variables correlations

# In[23]:


make_corr_plot(train_anndata.obs, obs_quant_vars, 'Correlation Matrix of Quantitative Variables in the obs Training Data (spearman)', method='spearman')
make_corr_plot(train_anndata.obs, obs_quant_vars, 'Correlation Matrix of Quantitative Variables in the obs Training Data (pearson)', method='pearson')


# ### Independent gene expression variables correlations

# In[24]:


make_corr_plot(df, marker_names, 'Correlation Matrix of Marker Expression in the Training Data (spearman)', method='spearman')


# In[25]:


make_corr_plot(df, marker_names, 'Correlation Matrix of Marker Expression in the Training Data (pearson)', method='pearson')


# In[26]:


df[marker_names].describe()


# ### Further correlation analysis utils

# In[164]:


def logistic_regression_importance_ranking(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X=X, y=y)
    return (abs(clf.coef_)[0].argsort()[::-1]).tolist()

def linear_regression_importance_ranking(X, y):
    lrm = LinearRegression()
    lrm.fit(X, y)
    return (abs(lrm.coef_).argsort()[::-1]).tolist()

def mutual_information_importance_ranking_class(X, y):
    return (mutual_info_classif(X, y).argsort()[::-1]).tolist()

def mutual_information_importance_ranking_reg(X, y):
    return (mutual_info_regression(X, y).argsort()[::-1]).tolist()

def plot_2d(x, y, classes, xlab, ylab, title=None):
    r"""
    Plot a 2D scatter plot of the given x and y values, colored by the given classes.
    """
    unique_classes = set(classes)
    colors = colormaps.get_cmap('tab10')

    plt.figure(figsize=(8, 8))
    for i, class_label in enumerate(unique_classes):
        class_indices = [j for j, c in enumerate(classes) if c == class_label]
        plt.scatter(x[class_indices], y[class_indices], color=colors(i), label=class_label, alpha=0.1, s=5)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_important_factors(factors, target, target_cat, focus_celltype=None, mutual_info_only=False):
    r"""
    Plot the most important factors based on logistic regression and mutual information.
    """
    factors = (factors - factors.mean(axis=0, keepdims=True)) / factors.std(axis=0, keepdims=True)
    if focus_celltype:
        target[target != focus_celltype] = 'Other'
    if target_cat:
        rank1 = logistic_regression_importance_ranking(factors, target)
        rank2 = mutual_information_importance_ranking_class(factors, target)
    else:
        rank1 = linear_regression_importance_ranking(factors, target)
        rank2 = mutual_information_importance_ranking_reg(factors, target)
    
    plot_2d(factors[:, rank2[0]], factors[:, rank2[1]], target, xlab=f"Factor {rank2[0]}", ylab=f"Factor {rank2[1]}", title='Factors selected with Mutual Information')
    if not mutual_info_only:
        plot_2d(factors[:, rank1[0]], factors[:, rank1[1]], target, xlab=f"Factor {rank1[0]}", ylab=f"Factor {rank1[1]}", title='Factors selected with Logistic Regression')

def plot_features_components_correlation(components, feature_names, correlation_threshold=0, title='', xlab='', display_desc=False):
    r"""
    Plot the correlation of the features with the components.
    """
    df_disp = pd.DataFrame(data=components.T * 100, index=feature_names, columns=list(range(1, 11)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_disp, annot=True, fmt=".0f", vmin=-100, vmax=100, linewidth=0.5, mask=abs(df_disp) < correlation_threshold, cmap=sns.diverging_palette(255, 10, as_cmap=True))
    if display_desc:
        plt.title(title)
    plt.xlabel(xlab)
    plt.show()


# ### PCA

# In[28]:


pca_on_scaled = PCA(n_components=10)
pca_on_unscaled = PCA(n_components=10)
pca_transformed_scaled_markers = pca_on_scaled.fit_transform(df[marker_names] / df[marker_names].std())
pca_transformed_unscaled_markers = pca_on_unscaled.fit_transform(df[marker_names])


# #### Components scatterplot - scaled markers

# In[29]:


plot_important_factors(pca_transformed_scaled_markers, targets_np.copy(), True, focus_celltype=None)


# In[30]:


plot_important_factors(pca_transformed_scaled_markers, targets_np.copy(), True, focus_celltype='Tumor')


# #### Components scatterplot - unscaled markers

# In[31]:


plot_important_factors(pca_transformed_unscaled_markers, targets_np.copy(), True, focus_celltype=None)


# In[32]:


plot_important_factors(pca_transformed_unscaled_markers, targets_np.copy(), True, focus_celltype='Tumor')


# #### Components importance analysis

# In[33]:


plt.plot(pca_on_scaled.explained_variance_ratio_, label='Scaled markers')
plt.plot(pca_on_unscaled.explained_variance_ratio_, label='Unscaled markers')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.show()


# In[34]:


plot_features_components_correlation(pca_on_scaled.components_, marker_names, correlation_threshold=20)


# In[35]:


plot_features_components_correlation(pca_on_unscaled.components_, marker_names, correlation_threshold=20)


# ### Factor Analysis

# In[36]:


fa_on_scaled = FactorAnalysis(n_components=10)
fa_on_unscaled = FactorAnalysis(n_components=10)
fa_transformed_scaled_markers = fa_on_scaled.fit_transform(df[marker_names] / df[marker_names].std())
fa_transformed_unscaled_markers = fa_on_unscaled.fit_transform(df[marker_names])


# #### Factor scatterplot - scaled markers

# In[37]:


plot_important_factors(fa_transformed_scaled_markers, targets_np.copy(), True, focus_celltype=None)


# In[38]:


plot_important_factors(fa_transformed_scaled_markers, targets_np.copy(), True, focus_celltype='Tumor')


# #### Factor scatterplot - unscaled markers

# In[39]:


plot_important_factors(fa_transformed_unscaled_markers, targets_np.copy(), True, focus_celltype=None)


# In[40]:


plot_important_factors(fa_transformed_unscaled_markers, targets_np.copy(), True, focus_celltype='Tumor')


# #### Components importance analysis

# In[41]:


plot_features_components_correlation(fa_on_scaled.components_, marker_names, correlation_threshold=20, title='Factor Analysis on Scaled Markers', xlab='Factors', display_desc=True)


# In[42]:


plot_features_components_correlation(fa_on_unscaled.components_, marker_names, correlation_threshold=20)


# ## 3. Intertype marker differentiation
# REQUIREMENT: Three biologically driven patterns of intertype marker differentiation (e.g., Tumor PDL1+ vs Tumor PDL1-, Mac CD206+ vs Mac CD206-, etc.),

# do a scatterplot colored by the PDL with the regression based factor selection as before but just for the selected types

# Other dimensionality reduction possibilities: Kernel PCA, UMAP, T-sne, self organizing maps
