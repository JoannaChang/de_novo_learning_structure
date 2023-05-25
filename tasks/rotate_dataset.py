#%%
import importlib
import numpy as np
from scipy.stats.stats import WeightedTauResult
import constants
import sys
import math
import create_dataset_toolbox as ct
importlib.reload(constants)
import os
importlib.reload(ct)

#%%
# set random seed
seed = int(sys.argv[1])
# seed = 100
print(seed)
np.random.seed(seed)

savdir = constants.Constants.PROJ_DIR + constants.Constants.DATA_FOLDER + str(seed) + '/'
if not os.path.exists(savdir):
    os.makedirs(savdir)

#%%
def create_rotated_dataset(orig_data:dict,angle:float,moveset:str, movnum:int = None, encoding:str='rad'):
    """ 
    Creates dataset with center-out reaches.
    For "uni" movesets: all trajectories reach towards/is rotated by a certain angle. Dataset corresponds to one movement.
    For "co" movesets: trajectories reach towards 8 targets. Dataset corresponds to 8 movements.

    Parameters
    ----------
    orig_data: original data (from 'create_dataset.py')
    angle: angle the trajectories should reach towards (in degrees)
    moveset: set of movements: uni (only one direction) or rot (8 targets, original centerout reach mov)
    movnum: number associated with mov
    encoding: 'rad', 'onehot'
    
    Returns
    ----------
    dic: dict
        transformed dataset
    """
    # transform training and test data
    dic = create_rotated_data(orig_data, angle, moveset, movnum = movnum, encoding= encoding)
    dic['test_set1'] = create_rotated_data(orig_data['test_set1'], angle, moveset, movnum = movnum, encoding= encoding)
    dic['test_set1']['params'] = dic['params']
    return dic

def rotate_uni_rad_stimulus(orig_data:dict, angle:float):
    """ 
    Creates rad (informative) stimuli for trajectories reaching towards a certain angle. 
    3D signal: (hold, cos(angle), sin(angle))

    Parameters
    ----------
    orig_data: original data (from 'create_dataset.py')
    angle: angle to reach to (in degrees)
    
    Returns
    ----------
    stimulus: array
    """
    orig_stim = orig_data['stimulus']
    stimulus = np.zeros((orig_stim.shape[0], orig_stim.shape[1], orig_stim.shape[2]))
    stimulus[:,:,0] = orig_stim[:,:,0] #copy hold signal
    angle_rad = math.radians(angle)

    for i in range(orig_stim.shape[0]):
        cue_onset = orig_data['cue_onset'][i]
        stimulus[i,cue_onset:,1] = 2*np.cos(angle_rad)
        stimulus[i,cue_onset:,2] = 2*np.sin(angle_rad)
    return stimulus

def rotate_uni_onehot_stimulus(orig_data:dict, movnum:int):
    """ 
    Creates onehot (uninformative) stimuli for trajectories reaching towards a certain angle. 
    5D signal: (hold, onehot encoding for 4 movements)

    Parameters
    ----------
    orig_data: original data (from 'create_dataset.py')
    movnum: number associated with movement in repertoire (0-3)
    
    Returns
    ----------
    stimulus: array
    """
    max_nmovs = 4
    orig_stim = orig_data['stimulus']
    stimulus = np.zeros((orig_stim.shape[0], orig_stim.shape[1], max_nmovs+1))
    stimulus[:,:,0] = orig_stim[:,:,0] #copy hold cue

    for i in range(orig_stim.shape[0]):
        cue_onset = orig_data['cue_onset'][i]
        stimulus[i,cue_onset:,movnum+1] = 2 #same magnitude as hold cue
    return stimulus

def rotate_centerout_onehot_stimulus(orig_data:dict):
    """ 
    Creates onehot (uninformative) stimuli for trajectories, based on experimental center-out reach with 8 targets. 
    9D signal: (hold, onehot encoding for 8 movements)

    Parameters
    ----------
    orig_data: original data (from 'create_dataset.py')
    
    Returns
    ----------
    stimulus: array
    """
    ntargets = 8
    orig_stim = orig_data['stimulus']
    stimulus = np.zeros((orig_stim.shape[0], orig_stim.shape[1], ntargets+1))
    stimulus[:,:,0] = orig_stim[:,:,0]
    target_id = orig_data['target_id']

    for i in range(orig_stim.shape[0]):
        cue_onset = orig_data['cue_onset'][i]
        stimulus[i,cue_onset:,target_id[i]+1] = 2 #same magnitude as hold cue
    return stimulus

def create_rotated_data(orig_data:dict, angle:float, moveset:str, movnum:int = None, encoding:str = 'rad'):
    """ 
    Creates data with all trajectories reaching towards/rotated by a certain angle. 
    Data could be training/test data.

    Parameters
    ----------
    orig_data: original data (from 'create_dataset.py')
    angle: angle the trajectories should reach towards (in degrees)
    moveset: set of movements: 
            uni (only one direction)
            co (center-out with 8 directions)
    movnum: number associated with mov
    encoding: 'rad', 'onehot'
    
    Returns
    ----------
    dic: dict
        transformed dataset
    """

    # rotate target
    angle_rad = math.radians(angle)
    rotation = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],[math.sin(angle_rad), math.cos(angle_rad)]])
    target = orig_data['target'] @ rotation # rotate clockwise

    # add mov encoding to stimulus
    if moveset == constants.Constants.UNI_MOVESET:
        nmovs = 1
        if encoding == 'onehot':
            assert (movnum is not None)
            stimulus = rotate_uni_onehot_stimulus(orig_data,movnum)
        elif encoding == 'rad':
            stimulus = rotate_uni_rad_stimulus(orig_data,angle)
        else:
            raise ValueError("unknown encoding")
    elif moveset == constants.Constants.CO_MOVESET:
        nmovs = 8
        if encoding == 'onehot':
            stimulus = rotate_centerout_onehot_stimulus(orig_data)
        elif encoding == 'rad':
            stimulus = orig_data['stimulus'] #keep original stimulus
        else:
            raise ValueError("unknown encoding")
    else:
        raise ValueError('unknown moveset')

    # get mov info
    target_param = np.array([x - angle_rad for x in orig_data['target_param']]) #dir in radians
    target_param = np.array([((angle + 2*math.pi) if angle < 0 else angle) for angle in target_param]) #only positive angles

    dic = {
        'go_onset': orig_data['go_onset'],
        'cue_onset': orig_data['cue_onset'],
        'idxoutofall': orig_data['idxoutofall'], 
        'target':target, #target positions
        'target_param': target_param, #actual direction
        'stimulus_param': target_param, #cued direction (diff for reassociation perturbation)
        'stimulus': stimulus, 
    }

    if 'params' in orig_data.keys():
        dic['params'] = orig_data['params']
        dic['params']['input_dim'] = stimulus.shape[2]
        dic['params']['output_dim'] = target.shape[2]
        dic['params']['moveset'] = moveset
        dic['params']['nmovs'] = nmovs
    
    return dic

def params_in_range(start_param:int, end_param:int, nmovs:int):
    """ Get evenly distributed n target parameters in a given range """
    inc = (end_param-start_param)/(nmovs-1)
    params = np.arange(start_param,end_param+inc,inc)
    params = np.round(params, decimals = 1)
    return params

def create_datasets_range(orig_data:dict, moveset:str, dir_pre:str, data_suffix:str, start_param:int, end_param:int, nmovs:int, ntrials:int, ntest_trials:int, encoding:str = 'rad', redo:bool = False):
    """ 
    Creates multi-mov datasets spanning range of movement params

    Parameters
    ----------
    orig_data: original data 
    moveset: set of movements: 
        uni (only one direction)
        co (8 targets, original centerout reach mov)
    dir_pre: directory where the data is saved
    data_suffix: suffix corresponding to dataset
    start_param: beginning of range of mov params
    end_param: end of range of mov params
    nmovs: number of movs to include in dataset
    ntrials: number of training trials
    ntest_trials: number of test trials
    'encoding': 'rad', 'onehot'
    'redo': whether to redo dataset if it already exists
    """

    # determine mov params
    params = params_in_range(start_param, end_param, nmovs)
    
    datasets = []
    onehot = (encoding == 'onehot')
    # create dataset for each mov param
    for i,param in enumerate(params):
        dir = dir_pre + str(param) + data_suffix +'.npy'
        if (not onehot) & os.path.exists(dir) & (not redo):
            datasets.append(np.load(dir,allow_pickle = True).item())
        else:
            dataset = create_rotated_dataset(orig_data, param, moveset, movnum = i, encoding = encoding)
            np.save(dir, dataset)
            datasets.append(dataset)
    
    # combine datasets to create multi-mov dataset
    comb_datasets = ct.subset_merge_data (datasets, ntrials, ntest_trials)
    dataset_name = '_'.join([str(nmovs)+'movs',str(start_param), str(end_param)])
    np.save(dir_pre+dataset_name+ data_suffix +'.npy', comb_datasets)


#%%
ntrials = 2016
ntesttrials = 192

# %%
# UNI DATASETS
#use uni-target dataset based on experimental reaches
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni.npy',allow_pickle = True).item()
moveset = constants.Constants.UNI_MOVESET
repertoire_pre = 'uni_'

## stimulus with 2D cue: cos(angle), sin(angle)
data_suffix = '_rad' 
encoding = 'rad'
### movements spanning clockwise
for nmovs in [2,3,4]:
    for end_param in [30,50,70,90]:
        create_datasets_range(orig_data, moveset, savdir+repertoire_pre, data_suffix, 
                        10, end_param, nmovs, ntrials, ntesttrials, encoding=encoding)
### movements spanning counterclockwise
for nmovs in [2,3,4]:
    for start_param in [-10,-30,-50]:
        create_datasets_range(orig_data, moveset, savdir+repertoire_pre, data_suffix, 
                        start_param, 10, nmovs, ntrials, ntesttrials, encoding=encoding)
### reaches from -30 to 90 to test generalization
create_datasets_range(orig_data, moveset, savdir+repertoire_pre, data_suffix, 
                        -30, 90, 7, ntrials, ntesttrials, encoding=encoding)

## stimulus with one-hot encoded cue for each target
data_suffix = '_onehot' 
encoding = 'onehot'
for nmovs in [2,3,4]:
    for end_param in [30,50,70,90]:
        create_datasets_range(orig_data, moveset, savdir+repertoire_pre, data_suffix, 
                        10, end_param, nmovs, ntrials, ntesttrials, encoding=encoding)

#%%
# use uni-target dataset based on synthetic reaches
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni_synth.npy',allow_pickle = True).item()
moveset = constants.Constants.UNI_MOVESET
repertoire_pre = 'uni_'

## stimulus with 2D cue: cos(angle), sin(angle)
data_suffix = '_synth_rad' 
encoding = 'rad'
for nmovs in [2,3,4]:
    for end_param in [30,50,70,90]:
        create_datasets_range(orig_data, moveset, savdir+repertoire_pre, data_suffix, 
                        10, end_param, nmovs, ntrials, ntesttrials, encoding=encoding)

#%%
# CENTEROUT DATASETS
moveset =constants.Constants.CO_MOVESET
#use original center-out dataset
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_centerout.npy',allow_pickle = True).item()
dataset = create_rotated_dataset(orig_data, angle = 0, moveset = moveset, encoding='rad')
np.save(savdir +'centerout_rad.npy', dataset)

#use original center-out dataset with onehot encoding
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_centerout.npy',allow_pickle = True).item()
dataset = create_rotated_dataset(orig_data, angle = 0, moveset = moveset, encoding='onehot')
np.save(savdir +'centerout_onehot.npy', dataset) 

# %%
# import matplotlib.pyplot as plt
# import constants
# importlib.reload(constants)
# import analysis.tools.output as ot
# importlib.reload(ot)

# seed = 100
# repertoires = constants.Constants.UNIS
# moveset = ot.get_moveset(repertoires[0])
# dataset = '_vel_rad'

# trial_index = 100
# repertoire = 'uni_4movs_10_50'
# data = ot.get_repertoire_data(dataset, repertoire, seed)
# stimulus = data['stimulus']
# target = data['target']

# fig, axs = plt.subplots(nrows = 2)
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

# colormap = ot.get_colormap(data['target_param'])
# plt.figure()
# for i in range(target.shape[0]):
#     plt.plot(target[i,:,0],target[i,:,1],  label = i, c = colormap[data['target_param'][i]])
