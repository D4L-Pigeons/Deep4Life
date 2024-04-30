#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


from IPython.display import clear_output; token = input(); clear_output()


# In[2]:


get_ipython().system(' git clone https://$token@github.com/SzymonLukasik/Deep4Life.git')


# In[3]:


get_ipython().run_line_magic('cd', '/content/Deep4Life')


# In[8]:


get_ipython().system('pip install anndata')


# In[6]:


get_ipython().system(' pip install pyometiff')


# ## Imports

# In[51]:


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

from typing import List
from src.datasets import load_d4ls
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[10]:


get_ipython().system('mkdir data')


# In[11]:


get_ipython().system('gdown 1-0YOHE1VoTRWqfBJLHQorGcHmkhCYvqW')


# In[12]:


load_d4ls.DATA_PATH


# In[ ]:


get_ipython().system('unzip train.zip -d $load_d4ls.DATA_PATH')


# ## Load anndata

# In[14]:


train_anndata = load_d4ls.load_full_anndata()


# In[15]:


image_name = train_anndata.obs.iloc[2]['image']
image_path = os.path.join(load_d4ls.TRAIN_IMAGE_DATA_IMAGES, image_name)
mask_path = os.path.join(load_d4ls.TRAIN_IMAGE_DATA_MASKS, image_name)

image_reader = pyometiff.OMETIFFReader(fpath=image_path)
mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)

image_array, _, _ = image_reader.read()
mask_array, _, _ = mask_reader.read()

# mask has 1 channel with values higher than 255 - each pixel is an object_id
# to properly visualize it we need to apply colormap

def apply_colormap(mask_array: np.ndarray) -> np.ndarray:
    # unique_ids = np.unique(mask_array)
    # colormap = plt.cm.get_cmap('tab20', len(unique_ids))
    # colored_mask = colormap(mask_array)

    colored_mask = mask_array / np.max(mask_array)
    return colored_mask

colored_mask = apply_colormap(mask_array)

plt.figure(figsize=(10, 10))
plt.imshow(colored_mask)
# add legend
# plt.colorbar(ticks=np.linspace(0, 1, len(np.unique(mask_array))))


# In[16]:


plt.imshow(image_array[2])


# In[103]:


image_array.shape


# In[104]:


mask_array.shape


# In[118]:


# check shapes of all images and masks
image_shapes = set()
mask_shapes = set()

for image_name in train_anndata.obs['image'].unique():
    image_path = os.path.join(load_d4ls.TRAIN_IMAGE_DATA_IMAGES, image_name)
    mask_path = os.path.join(load_d4ls.TRAIN_IMAGE_DATA_MASKS, image_name)

    image_reader = pyometiff.OMETIFFReader(fpath=image_path)
    mask_reader = pyometiff.OMETIFFReader(fpath=mask_path)

    image_array, _, _ = image_reader.read()
    mask_array, _, _ = mask_reader.read()

    image_shapes.add(image_array.shape)
    mask_shapes.add(mask_array.shape)


# In[119]:


image_shapes, mask_shapes


# ## CellSighter

# In[17]:


get_ipython().system(' git clone https://$token@github.com/SzymonLukasik/CellSighter.git ../CellSighter')
get_ipython().system(' git checkout lukass/baseline_experiment')


# In[19]:


CELL_SIGHTER_PATH = Path(load_d4ls.DATA_PATH.parent.parent / 'CellSighter')


# In[20]:


get_ipython().run_line_magic('cd', '../CellSighter')


# In[21]:


get_ipython().system('mkdir -p baseline_experiment/CellTypes/cells')
get_ipython().system('mkdir -p baseline_experiment/CellTypes/cells2labels')
get_ipython().system('mkdir -p baseline_experiment/CellTypes/data/images')


# In[22]:


get_ipython().system(' cp ../Deep4Life/data/train/images_masks/img/* baseline_experiment/CellTypes/data/images/')


# In[23]:


get_ipython().system(' cp ../Deep4Life/data/train/images_masks/masks/* baseline_experiment/CellTypes/cells/')


# In[47]:


len(train_anndata.var["marker"].unique()), len(train_anndata.var["channel"].unique())


# In[48]:


# save to txt file
train_anndata.var["marker"].to_csv("baseline_experiment/CellTypes/channels.txt", index=False, header=False)


# ### Create cell2labels

# In[24]:


train_anndata.obs["cell_labels"]


# In[25]:


len(train_anndata.obs["sample_id"].unique())


# In[26]:


# convert labels to numpy array of numbers (train_anndata.obs["cell_labels"] contains strings)
# you can use label_encoder from sklearn to convert strings to numbers

label_encoder = LabelEncoder()
label_encoder.fit(train_anndata.obs["cell_labels"])
cell_labels = label_encoder.transform(train_anndata.obs["cell_labels"])
cell_labels.shape


# In[27]:


236791 / 125


# In[28]:


# generate cell2labels

# group by sample_id and get dict of object_id -> cell_label
cell2labels =  train_anndata.obs.groupby("sample_id").apply(lambda x: {object_id: cell_label for object_id, cell_label in zip(x["ObjectNumber"], x["cell_labels"])}).to_dict()

cell2labels_encoded = {sample_id: {object_id: label_encoder.transform([cell_label])[0] for object_id, cell_label in cell_labels.items()} for sample_id, cell_labels in cell2labels.items()}


# In[29]:


sum(map(lambda x: len(x), cell2labels_encoded.values()))


# In[30]:


# for each sample in cell2labels_encoded create a dataframe with object_id as index and -1 as cell_label for non-existing objects

cell2labels_dfs = {sample_id: pd.DataFrame.from_dict(cell_labels, orient="index", columns=["cell_label"]).reindex(np.arange(1, len(cell_labels)).astype(int), fill_value=-1) for sample_id, cell_labels in cell2labels_encoded.items()}
list(cell2labels_dfs.values())[0]


# In[31]:


list(list(cell2labels.values())[0].items())[:5]


# In[40]:


# dict of label to cell type
hierarchy_match_dict = {
    id: cell_type for id, cell_type in enumerate(label_encoder.classes_)
}

hierarchy_match_dict


# In[32]:


# save cell2labels_dfs to files in a txt format (*.txt) where each line is separated by 


for sample_id, cell2labels_df in cell2labels_dfs.items():
    cell2labels_df.to_csv(f"baseline_experiment/CellTypes/cells2labels/{sample_id}.txt", sep="
", header=False, index=False)


# In[33]:


get_ipython().system(' ls baseline_experiment/CellTypes/cells2labels/* | wc -l')


# In[34]:


get_ipython().system(' ls baseline_experiment/CellTypes/cells/* | wc -l')


# In[35]:


get_ipython().system(' ls baseline_experiment/CellTypes/data/* | wc -l')


# In[68]:


# do the train_test split

X_train, X_test = train_test_split(train_anndata.obs["sample_id"].astype(str).unique(), test_size=0.2, random_state=42)


# In[44]:


len(X_train), len(X_test)


# In[53]:


example_config_path = "/content/CellSighter/example_experiment/cell_classification/config.json"
baseline_config_path = "/content/CellSighter/baseline_experiment/config.json"
with open(example_config_path) as f:
    config = json.load(f)
config


# In[155]:


new_config = {
    "crop_input_size":60,
    "crop_size":128,
    "root_dir":"./baseline_experiment/",
    "train_set": list(X_train[:8]),
    "val_set": list(X_test[:2]),
    "num_classes":14,
    "epoch_max":50,
    "lr":0.008,
    "blacklist":[
    ],
    "batch_size":64,
    "num_workers":2,
    "channels_path":"./baseline_experiment/CellTypes/channels.txt",
    "weight_to_eval":"",
    "sample_batch": "true",
    "to_pad": "false",
    "hierarchy_match": hierarchy_match_dict,
    # "size_data":10,
    "aug": "true"
 }


# In[80]:


get_ipython().system(' mkdir baseline_experiment/model_1')


# In[165]:


new_config_json_string = json.dumps(new_config)
with open("baseline_experiment/model_1/config.json", "w") as f:
    f.write(new_config_json_string)


# In[144]:


get_ipython().system('tensorboard --logdir=baseline_experiment/model_1')


# In[164]:


get_ipython().system(' git status')


# In[ ]:


get_ipython().system('python train.py --base_path=/content/CellSighter/baseline_experiment/model_1')

