#%%
import numpy as np 

class Repertoire_Dataset():
    """ Repertoire dataset"""
    def __init__(self, datadir, training =True):
        info = np.load(datadir+'.npy',allow_pickle = True).item()
        params = info['params']

        if training:  
            self.data = info
            self.ntrials = params['ntrials']
        else:
            self.data = info['test_set1']
            self.ntrials = len(self.data['target_param'])

        # target data
        self.output_dim = params['output_dim']
        self.target_output = self.data['target'][:,:,:self.output_dim] 

        # stimulus parameters from dataset 
        self.stimulus = self.data['stimulus']
        self.target_param = self.data['target_param'] #actual direction (radians) of trajectory
        self.tsteps = params['tsteps'] 
        self.input_dim = params['input_dim']
        self.dt = params['dt'] 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    datadir = "../data/dataset_chewie_uni"
    dataset = Repertoire_Dataset(datadir)
    # print(dataset.params)
    
    fig, axs = plt.subplots(nrows =2, figsize = (5,5), sharex=True)
    i = 2
    n_in = dataset.stimulus.shape[-1]
    #stimuli
    for j in range(n_in):
        axs[0].plot(dataset.stimulus[i,:,j], label= j) #hold
    # axs[0].plot(dataset.stimulus[i,:,1], label = 'cos', linestyle = '--') #cos
    # axs[0].plot(dataset.stimulus[i,:,2], label = 'sin', linestyle = 'dotted') #sin
    # axs[0].plot(dataset.stimulus[i,:,3], label = 'length', linestyle = '-.') #length
    axs[0].legend(title = 'Stimuli', bbox_to_anchor=(1.1, 1.05))

    #target: vel
    axs[1].plot(dataset.target_output[i,:,0], label = 'x pos') #x vel
    axs[1].plot(dataset.target_output[i,:,1], label = 'y pos') #y vel
    axs[1].legend(title = 'Target pos',  bbox_to_anchor=(1.1, 1.05))

    




    
# %%
