#%%
import numpy as np
import constants
import math

#%%
def sig(x,beta):
    """ Sigmoid function """
    return 1/(1+np.exp(-x*beta))

def create_target_data(start_points: np.array, end_points: np.array, go_onsets: np.array,vel: int, tsteps: int, output_dim: int):
    """ 
    Create target data for synthetic movements. Go onsets are based on experimental data.
    
    Parameters
    ----------
    start_points: array (ntrials,2) indicating the start points of the movements
    end_points: array (ntrials,2) indicating the end points of the movements
    go_onsets: array (ntrials) indicating the go onsets of the movements
    vel: velocity of the movement
    tsteps: number of time steps
    output_dim: output dimension

    """
    ntrials = start_points.shape[0]
    
    # prepare xaxis for smooth transition
    tsteps_movement = 100
    xx = np.linspace(-1,1,tsteps_movement,endpoint=False)
    ytemp = sig(xx,vel)
    
    # create target
    target = np.zeros((ntrials,tsteps,output_dim))
    for j in range(ntrials):
        go_onset = go_onsets[j]
        target[j,:go_onset,:2] = start_points[j]
        target[j,go_onset:,:2] = end_points[j]  
        target[j,go_onset:(go_onset+tsteps_movement),:2] = \
                ytemp[:,None]*(end_points[j]-start_points[j])[None,:]
        
    return target

def create_fixed_stimulus(orig_stim: np.array, cue_onset: int, go_onset: int):
    """ 
    Creates fixed stimuli for trajectories reaching towards 0 degrees. 
    3D signal: (hold, cos(angle), sin(angle))

    Parameters
    ----------
    orig_stim: original stimulus (ntrials, tsteps, ninputs)
    cue_onset: cue onset
    go_onset: go onset
    
    Returns
    ----------
    stimulus: array
    """
    stimulus = np.zeros((orig_stim.shape[0], orig_stim.shape[1], orig_stim.shape[2]))
    stimulus[:,:go_onset,0] = 2
    stimulus[:,cue_onset:,1] = 2*np.cos(0)
    stimulus[:,cue_onset:,2] = 2*np.sin(0)

    return stimulus

def create_synthetic_data(data: dict, vel: int, tsteps: int, output_dim: int, output_range: int, fixed_onsets: bool = False):
    """ 
    Create dataset for synthetic movements. 

    Parameters
    ----------
    data: dictionary containing the original dataset
    vel: velocity of the movement
    tsteps: number of time steps
    output_dim: output dimension
    output_range: range of the output (length of movement)
    fixed_onsets: whether to use fixed onsets for the movement
    
    """
    ntrials = len(data['target_id'])
    phis = np.repeat(0, ntrials)

    #start/end points and go-onsets for the movement
    start_points = np.zeros((ntrials,2))
    end_points = (output_range*np.array([np.cos(phis),np.sin(phis)])).T

    go_onsets = data['go_onset']

    if fixed_onsets:
        cue_onsets = data['cue_onset']
        cue_onsets = np.repeat(int(np.mean(cue_onsets)), ntrials)
        go_onsets = np.repeat(int(np.mean(go_onsets)), ntrials)

        data['go_onset'] = go_onsets
        data['cue_onset'] = cue_onsets
        data['stimulus'] = create_fixed_stimulus(data['stimulus'], cue_onsets[0], go_onsets[0])

    data['target'] = create_target_data(start_points, end_points, go_onsets, vel, tsteps, output_dim)

    return data


#%%
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni.npy',allow_pickle = True).item()
output_range = 8
vel = 6

tsteps = orig_data['params']['tsteps']
output_dim = orig_data['params']['output_dim']

fixed_synth_dataset = create_synthetic_data(orig_data, vel, tsteps, output_dim, output_range, fixed_onsets = True)
fixed_synth_dataset['test_set1'] = create_synthetic_data(orig_data['test_set1'], vel, tsteps, output_dim, output_range, fixed_onsets=True)

dir = constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni_synth_fixed.npy'
np.save(dir, fixed_synth_dataset)
# %%
