# %%
import numpy as np
from constants import Constants
import create_dataset_toolbox as ct
import math
import os
import sys
import matplotlib.pyplot as plt


# %%

def create_sinewave(amplitude: float, tsteps: int):
    """ Create sine wave with given parameters """
    x = np.linspace(0, 2*np.pi, tsteps)
    y = amplitude * np.sin(FREQUENCY*x)
    return y

def create_cosinewave(amplitude: float, tsteps: int):
    """ Create cosine wave with given parameters """
    x = np.linspace(0, 2*np.pi, tsteps)
    y = amplitude * np.cos(FREQUENCY*x)
    return y

def create_target_data(ampA: float, ampB: float, go_onsets: np.array, tsteps: int):
    """ 
    Create target data for synthetic movements. Go onsets are based on experimental data.
    
    Parameters
    ----------
    ampA: amplitude of sine wave
    ampB: amplitude of cosine wave
    go_onsets: go onsets for each trial
    tsteps: number of time steps

    Returns
    ----------
    target: np.array (ntrials, tsteps, 2)
        target data for synthetic movements 

    """
    ntrials = len(go_onsets)

    # prepare xaxis for smooth transition
    wave1 = create_sinewave(ampA, TSTEPS_MOVEMENT)
    wave2 = create_cosinewave(ampB, TSTEPS_MOVEMENT)

    # create target
    target = np.zeros((ntrials, tsteps, 2))

    for j in range(ntrials):
        go_onset = go_onsets[j]
        target[j, :go_onset, :] = 0

        # sine wave
        target[j, go_onset:, 0] = 0 #wave1[-1]
        target[j, go_onset:(go_onset+TSTEPS_MOVEMENT), 0] = wave1

        # cosine wave
        target[j, go_onset:, 1] = 0 #wave2[-1]
        target[j, go_onset:(go_onset+TSTEPS_MOVEMENT), 1] = wave2

    return target

def create_cont_onehot_stimulus(go_onsets: np.array, cue_onsets: np.array, tsteps: int, ampsA: np.array, movnums: np.array, max_nmovs=4):
    """ 
    Create stimulus data for synthetic movements. The sine wave is encoded with continuous stimulus, the cosine wave as one-hot.
    
    Parameters
    ----------
    go_onsets: go onsets for each trial
    cue_onsets: cue onsets for each trial
    tsteps: number of time steps
    ampsA: amplitude of sine wave for each trial
    movnums: movement number of cosine wave for each trial
    max_nmovs: maximum number of movements

    Returns
    ----------
    stimulus: np.array (ntrials, tsteps, 2*max_nmovs + 1)
        stimulus data for synthetic movements 

    """
    ntrials = len(go_onsets)

    # create stimulus (hold, ampA, ampB_1, ampB_2, ampB_3, ampB_4)
    stimulus = np.zeros((ntrials, tsteps, 2*max_nmovs + 1))
    for i in range(ntrials):
        go_onset = go_onsets[i]
        cue_onset = cue_onsets[i]
        stimulus[i, :go_onset, 0] = 2

        # continuous stimulus for sine wave
        stimulus[i, cue_onset:, 1] = 2*ampsA[i]/MAX_AMPLITUDE

        # one-hot encoded stimulus for cosine wave
        stimulus[i, cue_onset:, 2 + movnums[i]] = 2

    return stimulus


def create_sinusoidal_data(ntrials: int, ampA: float, ampB: float, movnum: int, encoding: str, vary_ampA: bool=True, pert_param: float=None, max_nmovs=4):
    """ 
    Create dataset for synthetic movements. 

    Parameters
    ----------
    ntrials: number of trials
    ampA: amplitude of sine wave
    ampB: amplitude of cosine wave
    movnum: movement number of cosine wave
    encoding: how inputs are encoded
    vary_ampA: whether to vary ampA or ampB
    pert_param: perturbation parameter
    max_nmovs: maximum number of movements

    Returns
    ----------
    dic: dict
        dataset information
    
    """

    cue_onsets = np.repeat(80, ntrials).astype(int)
    go_onsets = np.repeat(130, ntrials).astype(int)

    # create target, perturb if necessary
    if pert_param is not None:
        if vary_ampA:
            target = create_target_data(pert_param, ampB, go_onsets, TSTEPS)
        else:
            target = create_target_data(ampA, pert_param, go_onsets, TSTEPS)
    else:
        target = create_target_data(ampA, ampB, go_onsets, TSTEPS)

    # create stimulus
    if encoding == 'cont_onehot':
        ampsA = np.repeat(ampA, ntrials)
        ampB_movnum = 0 if vary_ampA else movnum
        ampB_movnums = np.repeat(ampB_movnum, ntrials)
        stimulus = create_cont_onehot_stimulus(
            go_onsets, cue_onsets, TSTEPS, ampsA, ampB_movnums, max_nmovs)
    else:
        raise ValueError('Encoding not recognized')

    # target params
    target_params = np.repeat(
        ampA, ntrials) if vary_ampA else np.repeat(ampB, ntrials)

    dic = {
        'go_onset': go_onsets,
        'cue_onset': cue_onsets,
        'idxoutofall': np.array(range(ntrials)),
        'target': target,  # target positions
        'target_param': target_params,  # actual parameter value
        # cued parameter value (diff for reassociation perturbation)
        'stimulus_param': target_params,
        'stimulus': stimulus,

    }

    dic['params'] = {}
    dic['params']['tsteps'] = TSTEPS
    dic['params']['ntrials'] = ntrials
    dic['params']['input_dim'] = stimulus.shape[2]
    dic['params']['output_dim'] = target.shape[2]
    dic['params']['dt'] = 0.01
    dic['params']['moveset'] = 'ampA' if vary_ampA else 'ampB'
    dic['params']['nmovs'] = 1
    dic['params']['pert_param'] = pert_param

    return dic


def create_sinusoidal_dataset(ntrials: int, ntest_trials: int, ampA: float, ampB: float, movnum: int, encoding: str, vary_ampA: bool=True, pert_param: float=None, max_nmovs: int=4):
    """ 
    Create dataset with training and testing data for synthetic movements. 
    
    Parameters
    ----------
    ntrials: number of trials
    ntest_trials: number of test trials
    ampA: amplitude of sine wave
    ampB: amplitude of cosine wave
    movnum: movement number of cosine wave
    encoding: how inputs are encoded
    vary_ampA: whether to vary ampA or ampB
    pert_param: perturbation parameter
    max_nmovs: maximum number of movements
    
    Returns
    ----------
    dic: dict
        dataset information
    """

    dic = create_sinusoidal_data(ntrials, ampA, ampB, movnum,
                               encoding, vary_ampA, pert_param, max_nmovs)
    dic['test_set1'] = create_sinusoidal_data(
        ntest_trials, ampA, ampB, movnum, encoding, vary_ampA, pert_param, max_nmovs)
    return dic


def params_in_range(start_param: int, end_param: int, nmovs: int):
    """ Get evenly distributed n target parameters in a given range """
    inc = (end_param-start_param)/(nmovs-1)
    params = np.arange(start_param, end_param+inc, inc)
    params = np.round(params, decimals=1)
    return params


def create_datasets_range(moveset: str, dir_pre: str, data_suffix: str, start_param: int, end_param: int, nmovs: int, ntrials: int, ntest_trials: int, encoding: str = 'cont', redo: bool = False, max_nmovs: int = 4):
    """ 
    Creates and saves multi-mov datasets spanning range of movement params

    Parameters
    ----------
    moveset: set of movements: 'ampA1', 'ampB1'
    dir_pre: directory where the data is saved
    data_suffix: suffix corresponding to dataset
    start_param: beginning of range of mov params
    end_param: end of range of mov params
    nmovs: number of movs to include in dataset
    ntrials: number of training trials
    ntest_trials: number of test trials
    encoding: 'cont', 'onehot'
    redo: whether to redo dataset if it already exists
    max_nmovs: maximum number of movements

    """

    # determine mov params
    params = params_in_range(start_param, end_param, nmovs)
    # TODO: fix this depending on moveset naming
    vary_ampA = ('ampA' in moveset)

    datasets = []
    # create dataset for each mov param
    for i, param in enumerate(params):
        dir = dir_pre + str(param) + data_suffix + '.npy'
        if os.path.exists(dir) & (not redo):
            datasets.append(np.load(dir, allow_pickle=True).item())
        else:
            if vary_ampA:
                ampA = param
                ampB = MIN_AMPLITUDE
            else:
                ampA = MIN_AMPLITUDE
                ampB = param
            dataset = create_sinusoidal_dataset(
                ntrials, ntest_trials, ampA, ampB, i, encoding, vary_ampA=vary_ampA,  max_nmovs=max_nmovs)
            np.save(dir, dataset)
            datasets.append(dataset)

    # return datasets
    # combine datasets to create multi-mov dataset
    comb_datasets = ct.subset_merge_data(datasets, ntrials, ntest_trials)
    dataset_name = '_'.join(
        [str(nmovs)+'movs', str(start_param), str(end_param)])

    np.save(dir_pre+dataset_name + data_suffix + '.npy', comb_datasets)

def create_cont_onehot_reassoc_dataset(vary_ampA_data: dict, vary_ampB_data: dict, max_nmovs: int=4):
    """ 
    Creates dataset with reassociated targets. 
    E.g. stimulus 1 originally associated with target 1 must now produce target 2.

    Parameters
    ----------
    vary_ampA_data: original dataset where amplitude of sine wave is varied
    vary_ampB_data: original dataset where amplitude of cosine wave is varied
    max_nmovs: maximum number of movements

    Returns
    ----------
    vary_ampA_data: dict
        data with reassociated targets for ampA
    vary_ampB_data: dict
        data with reassociated targets for ampB
    """
    # make sure same reassociations are used for both cont and onehot datasets
    nmovs = vary_ampA_data['params']['nmovs']

    # reassociate targets and mov numbers
    _, idx = np.unique(vary_ampA_data['stimulus_param'], return_index=True)
    stim_param_orig = vary_ampA_data['stimulus_param'][np.sort(idx)]

    movnum_old = np.arange(nmovs)
    movnum_new = movnum_old

    while (movnum_new == movnum_old).any():
        movnum_new = np.random.permutation(movnum_old)

    # dictionaries with new associations: use same reassociations for cont and onehot datasets
    movnum_reassoc = {stim_param_orig[i]: movnum_new[i] for i in range(nmovs)}
    stim_param_reassoc = {
        stim_param_orig[i]: stim_param_orig[movnum_new[i]] for i in range(nmovs)}
    # print(stim_param_reassoc)

    # create reassociated data for training and test data
    vary_ampA_data = reassoc_data(
        vary_ampA_data, 'cont_onehot', movnum_reassoc, stim_param_reassoc, vary_ampA= True, max_nmovs=max_nmovs)
    vary_ampA_data['test_set1'] = reassoc_data(
        vary_ampA_data['test_set1'], 'cont_onehot', movnum_reassoc, stim_param_reassoc, vary_ampA= True, max_nmovs=max_nmovs)
  
    vary_ampB_data = reassoc_data(
        vary_ampB_data, 'cont_onehot', movnum_reassoc, stim_param_reassoc, vary_ampA= False, max_nmovs=max_nmovs)
    vary_ampB_data['test_set1'] = reassoc_data(
        vary_ampB_data['test_set1'], 'cont_onehot', movnum_reassoc, stim_param_reassoc, vary_ampA= False, max_nmovs=max_nmovs)

    return vary_ampA_data, vary_ampB_data

def reassoc_data(data: dict, encoding: str, movnum_reassoc: dict, stim_params_reassoc: dict, vary_ampA: bool = True, max_nmovs: int = 4):
    """ 
    Creates data with reassociated targets. 
    E.g. stimulus 1 originally associated with target 1 must now produce target 2.

    Parameters
    ----------
    data: original data 
    dataset_type: type of dataset (cont or onehot)
    movnum_reassoc: original mov param : new mov numbers
    targets_reassoc: original mov param : new mov param
    stim_params_reassoc: original mov target: new mov target

    Returns
    ----------
    data: dataframe
        data with reassociated targets
    """

    # get new stimuli
    stim_params_new = [stim_params_reassoc[stim_param]
                       for stim_param in data['stimulus_param']]
    go_onsets = data['go_onset']
    cue_onsets = data['cue_onset']

    # reassociate the stimuli
    if encoding == 'cont_onehot':
        if vary_ampA:
            ampsA = stim_params_new
            ampB_movenums = np.repeat(0, len(go_onsets))
            data['stimulus'] = create_cont_onehot_stimulus(
                go_onsets, cue_onsets, TSTEPS, ampsA, ampB_movenums, max_nmovs)
        else:
            ampsA = np.repeat(MIN_AMPLITUDE, len(go_onsets))
            ampB_movnums = [movnum_reassoc[stim_param]
                       for stim_param in data['stimulus_param']]
            data['stimulus'] = create_cont_onehot_stimulus(
                go_onsets, cue_onsets, TSTEPS, ampsA, ampB_movnums, max_nmovs)
    else:
        raise ValueError('unknown dataset')

    # update stim_param, but keep mov_param to match target output
    data['stimulus_param'] = stim_params_new
    data['params']['reassoc_mapping'] = stim_params_reassoc

    return data


# %%
seed = int(sys.argv[1])
# seed = 1000000
print(seed)
np.random.seed(seed)
redo = True

savdir = Constants.PROJ_DIR + Constants.DATA_FOLDER + str(seed) + '/'
if not os.path.exists(savdir):
    os.makedirs(savdir)

# %%
FREQUENCY = 3
MIN_AMPLITUDE = 1
MAX_AMPLITUDE = 7

# perturbation
PERT_AMPLITUDE = 2  # fixed from 2

# trial info
TSTEPS = 400
TSTEPS_MOVEMENT = 200
ntrials = 120
ntest_trials = 120
max_nmovs = 4

# moveset info
suffix = ''
cont_onehot_suffix = suffix + '_cont_onehot'

ampA_moveset = Constants.AMPA1_MOVESET
ampB_moveset = Constants.AMPB1_MOVESET

# %%
# AMPA DATASETS
moveset = ampA_moveset
repertoire_pre = ampA_moveset+'_'

# stimulus with continuous cue for ampA, and one-hot encoded cue for ampB
data_suffix = cont_onehot_suffix
encoding = 'cont_onehot'
for nmovs in range(2, max_nmovs+1):
    create_datasets_range(moveset, savdir+repertoire_pre, data_suffix, MIN_AMPLITUDE, MAX_AMPLITUDE,
                          nmovs, ntrials, ntest_trials, encoding=encoding, redo=redo, max_nmovs=max_nmovs) 

# AMPB DATASETS
moveset = ampB_moveset
repertoire_pre = ampB_moveset+'_'

# stimulus with continuous cue for ampA, and one-hot encoded cue for ampB
data_suffix = cont_onehot_suffix
encoding = 'cont_onehot'
for nmovs in range(2, max_nmovs+1):
    create_datasets_range(moveset, savdir+repertoire_pre, data_suffix, MIN_AMPLITUDE, MAX_AMPLITUDE,
                          nmovs, ntrials, ntest_trials, encoding=encoding, redo=redo, max_nmovs=max_nmovs)

# %%
# FOR PERTURBATION
# pert ampA
repertoire_pre = ampA_moveset+'_'
for encoding in ['cont_onehot']:
    data_suffix = f'{suffix}_{encoding}_{Constants.PERT_AMPLITUDE_A}'
    dir = savdir+repertoire_pre + \
        str(float(MIN_AMPLITUDE)) + data_suffix + '.npy'
    dataset = create_sinusoidal_dataset(ntrials, ntest_trials, MIN_AMPLITUDE, MIN_AMPLITUDE, 0, encoding,
                                      vary_ampA=True, pert_param=PERT_AMPLITUDE, max_nmovs=max_nmovs)
    np.save(dir, dataset)

# pert ampB
repertoire_pre = ampB_moveset+'_'
for encoding in ['cont_onehot']:
    data_suffix = f'{suffix}_{encoding}_{Constants.PERT_AMPLITUDE_B}'
    dir = savdir+repertoire_pre + \
        str(float(MIN_AMPLITUDE)) + data_suffix + '.npy'
    dataset = create_sinusoidal_dataset(ntrials, ntest_trials, MIN_AMPLITUDE, MIN_AMPLITUDE, 0, encoding,
                                      vary_ampA=False, pert_param=PERT_AMPLITUDE,  max_nmovs=max_nmovs)
    np.save(dir, dataset)

# create reassociated dataset based on existing, preprocessed dataset, for cont_onehot
for nmovs in range(2, max_nmovs+1):
    dataset_name = '_'.join(
        [str(nmovs)+'movs', str(MIN_AMPLITUDE), str(MAX_AMPLITUDE)])

    repertoire_pre = f'{ampA_moveset}_'
    ampA_datadir_pre = savdir+repertoire_pre
    vary_ampA_data = np.load(
        ampA_datadir_pre+dataset_name+cont_onehot_suffix + '.npy', allow_pickle=True).item()

    repertoire_pre = f'{ampB_moveset}_'
    ampB_datadir_pre = savdir+repertoire_pre
    vary_ampB_data = np.load(
        ampB_datadir_pre+dataset_name+cont_onehot_suffix + '.npy', allow_pickle=True).item()

    vary_ampA_data, vary_ampB_data = create_cont_onehot_reassoc_dataset(
            vary_ampA_data, vary_ampB_data, max_nmovs=max_nmovs)
            
    np.save(ampA_datadir_pre+dataset_name +
            f'{cont_onehot_suffix}_{Constants.PERT_REASSOCIATION}.npy', vary_ampA_data)
    np.save(ampB_datadir_pre+dataset_name +
            f'{cont_onehot_suffix}_{Constants.PERT_REASSOCIATION}.npy', vary_ampB_data)

