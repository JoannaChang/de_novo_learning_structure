#%%
import importlib
import numpy as np
import constants
import os
import sys
import create_dataset_toolbox as ct
importlib.reload(constants)
importlib.reload(ct)

#%%
#set random seed
seed = int(sys.argv[1])
# seed = 100
print(seed)
np.random.seed(seed)

savdir = constants.Constants.PROJ_DIR + constants.Constants.DATA_FOLDER + str(seed) + '/'
if not os.path.exists(savdir):
    os.makedirs(savdir)

#%%
def create_reassoc_dataset(dir_pre:str, dataset_name:str):
    """ 
    Creates dataset with reassociated targets. 
    E.g. stimulus 1 originally associated with target 1 must now produce target 2.

    Parameters
    ----------
    dir_pre: directory prefix to store dataset
    dataset_name: name of dataset
    
    Returns
    ----------
    rad_data: dataframe
        dataset with informative stimuli about target direction
    onehot_data: dataframe
        dataset with uninformative, one-hot encoded stimuli
    """
    #load original data, make sure same reassociations are used for both rad and onehot datasets
    rad_data = np.load(dir_pre+dataset_name+'_rad'+'.npy', allow_pickle=True).item()
    onehot_data = np.load(dir_pre+dataset_name+'_onehot'+'.npy', allow_pickle=True).item()
    nmovs = rad_data['params']['nmovs']

    #reassociate targets and mov numbers
    _, idx = np.unique(rad_data['target_param'], return_index=True)
    targets_orig = rad_data['target_param'][np.sort(idx)]

    _, idx = np.unique(rad_data['stimulus_param'], return_index=True)
    stim_param_orig = rad_data['stimulus_param'][np.sort(idx)]

    movnum_old = np.arange(nmovs)
    movnum_new = movnum_old
    while (movnum_new == movnum_old).any():
        movnum_new  = np.random.permutation(movnum_old)

    ## dictionaries with new associations: use same reassociations for rad and onehot datasets
    movnum_reassoc = {targets_orig[i]: movnum_new[i] for i in range(nmovs)}
    targets_reassoc = {targets_orig[i]: targets_orig[movnum_new[i]] for i in range(nmovs)}
    stim_param_reassoc = {stim_param_orig[i]: stim_param_orig[movnum_new[i]] for i in range(nmovs)}

    #create reassociated data for training and test data
    rad_data = reassoc_uni(
        rad_data, 'rad', movnum_reassoc, targets_reassoc, stim_param_reassoc)
    rad_data['test_set1'] = reassoc_uni(
        rad_data['test_set1'], 'rad', movnum_reassoc, targets_reassoc, stim_param_reassoc)
        
    onehot_data = reassoc_uni(
        onehot_data, 'onehot', movnum_reassoc, targets_reassoc, stim_param_reassoc)
    onehot_data['test_set1'] = reassoc_uni(
        onehot_data['test_set1'], 'onehot', movnum_reassoc, targets_reassoc, stim_param_reassoc)

    return rad_data, onehot_data

#%%
def reassoc_uni(data:dict, dataset_type:str, movnum_reassoc:dict, targets_reassoc:dict, stim_params_reassoc:dict):
    """ 
    Creates data with reassociated targets. 
    E.g. stimulus 1 originally associated with target 1 must now produce target 2.

    Parameters
    ----------
    data: original data 
    dataset_type: type of dataset (rad or onehot)
    movnum_reassoc: original mov param : new mov numbers
    targets_reassoc: original mov param : new mov param
    stim_params_reassoc: original mov target: new mov target
    
    Returns
    ----------
    data: dataframe
        data with reassociated targets
    """

    # get new targets and mov targets
    targets_new = [targets_reassoc[target] for target in data['target_param']]
    stim_params_new = [stim_params_reassoc[stim_param] for stim_param in data['stimulus_param']]

    # reassociate the stimuli
    if dataset_type == 'rad':
        data['stimulus'] = reassoc_uni_rad_stimulus(data,targets_new)
    elif dataset_type == 'onehot':
        movnums_new = [movnum_reassoc[target_orig] for target_orig in data['target_param']]
        data['stimulus'] = reassoc_uni_onehot_stimulus(data,movnums_new)
    else:
        raise ValueError('unknown dataset')

    # update stim_param, but keep mov_param to match target output
    data['stimulus_param'] = stim_params_new
    data['params']['reassoc_mapping'] = targets_reassoc

    return data

def reassoc_uni_rad_stimulus(data:dict, targets_new:list):
    """ 
    Creates rad (informative) stimuli with reassociated targets. 

    Parameters
    ----------
    data: original data 
    targets_new: new mov params
    
    Returns
    ----------
    stimulus: array
        stimuli with reassociated targets
    """
    stimulus = data['stimulus']
    for i, cue_onset in enumerate(data['cue_onset']):
        angle_rad = -targets_new[i]
        stimulus[i,cue_onset:,1] = 2*np.cos(angle_rad)
        stimulus[i,cue_onset:,2] = 2*np.sin(angle_rad)
    return stimulus

def reassoc_uni_onehot_stimulus(data:dict, movnums_new:list):
    """ 
    Creates onehot (uninformative) stimuli with reassociated targets. 

    Parameters
    ----------
    data: original data 
    movnums_new: new mov numbers
    
    Returns
    ----------
    stimulus: array
        stimuli with reassociated targets
    """
    stimulus = data['stimulus']
    for i, cue_onset in enumerate(data['cue_onset']):
        movnum = movnums_new[i]
        stimulus[i,cue_onset:, 1:] = 0 #reset all movs to 0
        stimulus[i,cue_onset:,movnum+1] = 2 #same magnitude as hold cue
    return stimulus

#%%
#create reassociated dataset based on existing, preprocessed dataset
repertoire_pre = 'uni_'
dir_pre = savdir+repertoire_pre

start_param = 10
for nmovs in [2,3,4]:
    for end_param in [30,50,70,90]:
        dataset_name = '_'.join([str(nmovs)+'movs',str(start_param), str(end_param)])
        # print(dataset_name)
        rad_data, onehot_data = create_reassoc_dataset(dir_pre, dataset_name)
        np.save(dir_pre+dataset_name+ '_rad_' + constants.Constants.PERT_REASSOCIATION + '.npy', rad_data)
        np.save(dir_pre+dataset_name+ '_onehot_' + constants.Constants.PERT_REASSOCIATION +'.npy', onehot_data)

# %%
# Plot to check data
# import matplotlib.pyplot as plt
# import analysis.tools.output as ot

# trial_index = 10
# y_pos = 10
# data = rad_data
# stim_param = data['stimulus_param']
# target = data['target']
# plt.figure()
# colormap = ot.get_colormap(stim_param, 'viridis_r')
# for i in range(len(data['stimulus_param'])):
#     plt.plot(target[i,:,0],target[i,:,1], '-', c = colormap[stim_param[i]])
# plt.gca().set_aspect(1)
# plt.xlim([-y_pos-2,y_pos+2])
# plt.ylim([-y_pos-2,y_pos+2])

# fig, axs = plt.subplots(nrows = 2)
# stimulus = data['stimulus']
# for i in range(stimulus.shape[2]):
#     axs[0].plot(stimulus[trial_index,:,i], label = i)
# axs[0].vlines([data['cue_onset'][trial_index],data['go_onset'][trial_index]], 
#                 ymin = -2, ymax = 2, linestyles = 'dashed')
# axs[0].set_ylabel('stimulus')
# axs[0].legend()

# axs[1].plot(target[trial_index,:,0])
# axs[1].plot(target[trial_index,:,1])
# axs[1].vlines([data['cue_onset'][trial_index],data['go_onset'][trial_index]], 
#                 ymin = -2, ymax = 2, linestyles = 'dashed')
# axs[1].set_ylabel('position')


# %%
