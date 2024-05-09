#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from pathlib import Path

# Specify the directory path
directory = Path('../best_results/stellar')

method_accuracies = {}

# Iterate through each file in the directory
for item in directory.iterdir():
    # Check if the current item is a directory
    if item.is_dir():
        metric = pd.read_json(item / "test_metrics.json")
        method_accuracies[item.name] = [metric.iloc[-1]["accuracy"], metric.iloc[-1]["f1_score"]]
        

results = pd.DataFrame.from_dict(method_accuracies, orient="index", columns=["accuracy", "f1_score"])


# In[6]:


# Specify the directory path
directory = Path('../best_results/xgboost')

# Iterate through each file in the directory
for item in directory.iterdir():
    # Check if the current item is a directory
    if item.is_dir():
        metric = pd.read_json(item / "test_metrics.json")
        method_accuracies[item.name] = [metric.iloc[-1]["accuracy"], metric.iloc[-1]["f1_score"]]
        



# In[8]:


# Specify the directory path
directory = Path('../best_results/sklearn_mlp')

# Iterate through each file in the directory
for item in directory.iterdir():
    # Check if the current item is a directory
    if item.is_dir():
        metric = pd.read_json(item / "test_metrics.json")
        method_accuracies[item.name] = [metric.iloc[-1]["accuracy"], metric.iloc[-1]["f1_score"]]
    


# In[17]:


# Specify the directory path
directory = Path('../best_results/torch_mlp')

# Iterate through each file in the directory
# for item in directory.iterdir():
metric = pd.read_json(directory / "test_metrics.json")
# print(metric)
method_accuracies["torch_mlp"] = [metric.iloc[-1]["accuracy"], metric.iloc[-1]["f1_score"]]


# In[ ]:


# Specify the directory path
directory = Path('../best_results/sklearn_svm/svc')

# Iterate through each file in the directory
for item in directory.iterdir():
    # Check if the current item is a directory
    if item.is_dir():
        metric = pd.read_json(item / "test_metrics.json")
        method_accuracies[item.name] = [metric.iloc[-1]["accuracy"], metric.iloc[-1]["f1_score"]]
    


# In[18]:


results = pd.DataFrame.from_dict(method_accuracies, orient="index", columns=["accuracy", "f1_score"])
results


# In[4]:


ax = results.plot(kind='bar', y="accuracy")
plt.title('Column Plot Example')

# Adding labels with values at the top of each column
for index, value in enumerate(results["accuracy"]):
    ax.text(index, value + 0.001, str(value), ha='center', va='bottom')

plt.xlabel('Method')
plt.ylabel('Value')
plt.show()


# In[7]:


result_path = "..\\final_results\\xgboost\\standard_2024-05-08_20-50-46_seed_42_folds_5"


# In[13]:


metrics = pd.read_json(result_path + "\\metrics.json")


# In[29]:


metrics.iloc[-1]

