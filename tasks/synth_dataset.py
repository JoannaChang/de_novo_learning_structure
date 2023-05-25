#%%
import numpy as np
import constants

#%%
def sig(x,beta):
    """ Sigmoid function """
    return 1/(1+np.exp(-x*beta))

def create_target_data(start_points,end_points,go_onsets,vel,tsteps,output_dim):
    """ Create target data for synthetic movements. Go onsets are based on experimental data."""
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

def create_synthetic_data(data, vel, tsteps, output_dim, output_range):
    """ Create dataset for synthetic movements. """
    ntrials = len(data['target_id'])
    phis = np.repeat(0, ntrials)

    #start/end points and go-onsets for the movement
    start_points = np.zeros((ntrials,2))
    end_points = (output_range*np.array([np.cos(phis),np.sin(phis)])).T
    go_onsets = data['go_onset']

    data['target'] = create_target_data(start_points, end_points, go_onsets, vel, tsteps, output_dim)

    return data


#%%
orig_data = np.load(constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni.npy',allow_pickle = True).item()
output_range = 8
vel = 6

tsteps = orig_data['params']['tsteps']
output_dim = orig_data['params']['output_dim']
synth_dataset = create_synthetic_data(orig_data, vel, tsteps, output_dim, output_range)
synth_dataset['test_set1'] = create_synthetic_data(orig_data['test_set1'], vel, tsteps, output_dim, output_range)

dir = constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER + 'dataset_uni_synth.npy'
np.save(dir, synth_dataset)