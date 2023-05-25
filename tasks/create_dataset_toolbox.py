# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from constants import Constants
from typing import List

# %%
trial_nonspecific = ['params', 'test_set1', 'test_set2']

def subset_indices(d:dict,indices:list):
    """Subset data based on indices (trials)"""
    data = copy.deepcopy(d)
    for key in data.keys():
        if key not in trial_nonspecific:
            data[key] = data[key][indices,]
    return data

def merge_data (datasets:List[dict]):
    """Merge data from multiple datasets"""
    d_merged = dict.fromkeys(datasets[0].keys())
    for key in d_merged.keys():
            if key not in trial_nonspecific:
                d_merged[key] = np.concatenate([d[key] for d in datasets], axis=0)
    if 'params' in datasets[0].keys():
        d_merged['params'] = datasets[0]['params'].copy()
        d_merged['params']['ntrials'] = len(d_merged['target_param'])
        d_merged['params']['nmovs'] = len(datasets)
    return d_merged

def subset_merge_data (datasets:List[dict], total_ntrials:int, total_ntest_trials:int):
    """Subset and merge data from multiple datasets"""
    sub_ntrials = int(total_ntrials / len(datasets))
    sub_ntest_trials = int(total_ntest_trials / len(datasets))

    #subset trials from each dataset
    subset_data = [None] * len(datasets)
    for i in range(len(datasets)):
        subset_data[i] = subset_indices(datasets[i], get_target_indices(datasets[i]['stimulus_param'], sub_ntrials))
        subset_data[i]['test_set1'] = subset_indices(subset_data[i]['test_set1'], get_target_indices(subset_data[i]['test_set1']['stimulus_param'], sub_ntest_trials))

    # merge datasets
    merged_data = merge_data(subset_data)
    merged_data['test_set1'] = merge_data([d['test_set1'] for d in subset_data])

    return merged_data

def get_target_indices (stimulus_param:float, ntrials:int):
    """Get equal number of trials for each stimulus parameter"""
    unique_stim_params, unique_counts = np.unique(stimulus_param,return_counts=True)

    idx = []
    for stim_param in unique_stim_params:
        stim_param_indices = np.where(stimulus_param == stim_param)[0]
        n_indices = int(ntrials/len(unique_stim_params))
        indices = np.random.choice(stim_param_indices, n_indices, replace = False) 
        idx.extend(indices)
    return idx


# %%
