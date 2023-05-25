from urllib.parse import non_hierarchical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from matplotlib import cm
import matplotlib as mpl
from config_manager import base_configuration
from simulation.config_template import ConfigTemplate
import os
import contextlib
import math
from sklearn.decomposition import PCA
from typing import Callable, List



import seaborn as sns
import pyaldata as pyal

from tools.test import test_model
from constants import Constants
import torch
import scipy
import figure_params
figure_params.set_rc_params()

# PLOTTING ########################################################
def get_signif_annot(value:float):
    """ Get significance annotation for a given p-value. """

    assert(value >=0)

    if value >0.05:
        return 'n.s.'
    elif value > 0.01:
        return '$*$'
    elif value > 0.001:
        return '$*\,$*'
    else:
        return '$**$*'

def get_colormap(categories:np.array, cmap:str = 'plasma_r',  truncate:bool = True):
    """ 
    Get a colormap for a given set of categories. 

    Parameters
    ----------
    categories: array of categories
    cmap: matplotlib colormap
    truncate: whether to truncate the colormap
        
    """
    color_labels = np.unique(categories)
    if truncate:
        colors = [cm.get_cmap(cmap)(i) for i in np.linspace(0.1,0.9,len(color_labels))]
    else:
        colormap = cm.get_cmap(cmap, len(color_labels))
        colors = colormap(np.arange(0,len(color_labels)))
        
    color_map = dict(zip(color_labels, colors))
    return color_map

# GETTING DATA ###################################################
def get_moveset(repertoire:str):
    """ Get moveset for a given repertoire"""
    if repertoire in Constants.UNIS:
        return Constants.UNI_MOVESET
    elif repertoire in Constants.COS:
        return Constants.CO_MOVESET
    else:
        raise ValueError("Unknown moveset")

def get_repertoire_data(dataset:str, repertoire:str, seed:int):
    """ Get target data for a given repertoire. """
    datname = Constants.PROJ_DIR + Constants.DATA_FOLDER + str(seed) + '/' + repertoire + dataset
    data = np.load(datname+'.npy',allow_pickle = True).item()
    return data

def get_test_data(dataset:str, repertoire:str, seed:int):
    """ Get testing data for a given repertoire"""
    data = get_repertoire_data(dataset, repertoire,seed)
    return data['test_set1']

def get_train_data(outdir:str):
    """ Get data saved during training. Contains loss and parameters before/after training."""
    data = np.load(outdir+'training.npy',allow_pickle = True).item()
    return data

def get_outdir(seed:int, sim_set:str, repertoire:str, perturbation:str=None, pert_param:float=None, pert_set:str = None):
    """ 
    Get output directory of simulation. 

    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set 
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set 

    """

    outdir = Constants.PROJ_DIR + Constants.RESULTS_FOLDER + sim_set + '/' + str(seed) + "/" + repertoire +'/' 
    if perturbation:
        set = ('/' + pert_set) if pert_set else ""
        pert_dir = perturbation + set + '/'+ str(pert_param)+'/'
        outdir = outdir + pert_dir
    return outdir

# PLOTTING OUTPUT ###################################################
def graph_position (positions, ax, task_info, cmap= None, graph_all = False, **plot_kwargs):
    """
    Graph x,y positions for a given set of trials.

    Parameters
    ----------
    positions: trajectory positions
    ax: matplotlib axes
    task_info: task information
    cmap: colormap
    graph_all: whether to graph all trials or a random subset
    plot_kwargs: additional plotting arguments

    """

    # make color map based on different stimuli
    if cmap is not None:
        colormap = get_colormap(task_info, cmap = cmap) 
    else:
        if len(np.unique(task_info)) == 1:
           colormap = get_colormap(task_info,Constants.MOVEMENT_CMAP)
        else: 
            colormap = get_colormap(task_info,Constants.MOVEMENT_CMAP+'_r')

    y_pos = Constants.MAX_Y_POS

    #choose trials
    if graph_all:
        idx = range(len(task_info))
    else:
        #choose random trials to plot
        np.random.seed(1234)
        idx = []
        for target in np.unique(task_info):
            target_idx = np.random.choice(np.where(task_info == target)[0], 15)
            idx.extend(target_idx)

    positions = positions[idx]
    task_info = task_info[idx]
    params = np.unique(task_info)

    # graph trajectory positions
    for i in range(positions.shape[0]):
        ax.plot(positions[i,:,0],positions[i,:,1], '.', c = colormap[task_info[i]],
                linestyle = '-', linewidth = 1, marker = None, **plot_kwargs)

    # plot targets
    for param in params:
        ax.scatter(8*math.cos(param), 8*math.sin(param), 
               facecolors = 'none', edgecolors = colormap[param], s = 80, marker = 'o', linewidths = 1)

    ax.set_aspect(1)
    ax.set_xlim([-1,y_pos])
    ax.set_ylim([-y_pos,1])
    ax.set_axis_off()
    
def graph_repertoire (dataset:str, repertoire:str, seed:int = Constants.SEED_EX, ax:plt.Axes = None, cmap:str = None, graph_all:bool = False):
    """ 
    Graph x,y positions for repertoire based on training data.
    
    Parameters
    ----------
    dataset: training dataset
    repertoire: movement repertoire
    seed: random seed for simulation
    ax: matplotlib axes
    cmap: colormap
    graph_all: whether to graph all trials or a random subset
    
    """
    
    ax = ax or plt.gca()

    # get data
    data = get_test_data(dataset, repertoire, seed)

    target_param = data['target_param']
    target = data['target']
    task_info = np.array(target_param)

    graph_position(target, ax, task_info, cmap= cmap, graph_all = graph_all)
    return ax

def graph_output(seed:int, sim_set:str, repertoire:str, perturbation:str = None, pert_param:float = 0, pert_set:str=None, ax:plt.Axes = None, epoch:int = None, datafile:str =None, noise:float = None, graph_all:bool = False):
    """ 
    Graph x,y positions for given repertoire based on testing data.

    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    ax: matplotlib axes
    epoch: training epoch
    datafile: datafile to use for testing
    noise: noise level; if None, use noise level from training
    graph_all: whether to graph all trials or a random subset

    """

    ax = ax or plt.gca()

    outdir = get_outdir(seed, sim_set, repertoire, perturbation, pert_param, pert_set)
    datadir,_,output,_ = test_model(outdir, epoch, datafile = datafile, noise = noise)

    # get data
    datname = Constants.PROJ_DIR + datadir
    data = np.load(datname+'.npy',allow_pickle = True).item()['test_set1']    
    task_info = np.array(data['stimulus_param'])

    graph_position(output, ax, task_info, graph_all = graph_all)
    return ax


def graph_outputs_all(seed:int, sim_set:str, repertoires:List[str], perturbation:str=None, pert_param:float=0, pert_set:str=None, epoch:int=None, datafile:str=None, noise:float=None, graph_all:bool=False, learning_datafile:bool=False):
    """ 
    Graph x,y positions for all repertoires based on testing data.
    
    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set
    repertoires: movement repertoires
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    epoch: training epoch
    datafile: datafile to use for testing
    noise: noise level; if None, use noise level from training
    graph_all: whether to graph all trials or a random subset
    learning_datafile: whether to use datafile from skill learning

    """
    fig, axs = plt.subplots(ncols = (len(repertoires)), figsize = (2*len(repertoires),2))
    for i, repertoire in enumerate(repertoires):
        if learning_datafile:
            datafile = repertoire + Constants.SIM_SET_DATA[sim_set]
        graph_output(seed, sim_set, repertoire, perturbation, pert_param, pert_set = pert_set, 
            ax = axs[i], epoch = epoch, datafile = datafile, noise = noise, graph_all=graph_all)

    # save plot
    params = '_'.join(filter(None, [sim_set, str(seed), perturbation, pert_set, \
        str(pert_param) if pert_param else None, datafile, 
        'learning_datafile' if learning_datafile else None, str(epoch) if epoch else None]))
    return axs, params

# LOSS/MSE ###################################################
def get_loss(seed:int, sim_set:str, repertoire:str, perturbation:str=None, pert_param:float=None, pert_set:str = 'None', smooth:bool=False, smoothing_interval:int = 5, measure:str = 'loss'):
    """ 
    Get loss during training.

    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    smooth: whether to smooth loss
    smoothing_interval: number of epochs to smooth over
    measure: loss or test_error

    """
    outdir = get_outdir(seed, sim_set, repertoire, perturbation, pert_param, pert_set)
    train_data = get_train_data(outdir)
    if measure == 'loss':
        loss = list(train_data['lc'][:,0])
    elif measure == 'test_error':
        loss = list(train_data['lc'][:,1])
    else:
        raise ValueError('unknown measure')
    
    if smooth:
        # smooth loss based on past epochs only
        loss = uniform_filter1d(loss, size = smoothing_interval, origin = math.floor(smoothing_interval/2), mode='nearest')
    return loss

def get_loss_df(seeds:List[int], sim_set:str, repertoires:List[str], perturbation:str=None, pert_param:float=None, pert_set:str = 'None', smooth:bool=False, smoothing_interval:int = 5,  measure:str = 'loss'):
    """ 
    Get dataframe for loss during training.
    
    Parameters
    ----------
    seeds: random seeds for simulations
    sim_set: simulation set
    repertoires: movement repertoires
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    smooth: whether to smooth loss
    smoothing_interval: number of epochs to smooth over
    measure: loss or test_error

    """
    loss_df = pd.DataFrame()

    # get loss for each simulation and save in dataframe
    for i in range(len(repertoires)):
        repertoire = repertoires[i]
        for seed in seeds:
            loss = get_loss(seed, sim_set, repertoire, perturbation, pert_param, pert_set,
                            smooth, smoothing_interval, measure)
            repertoire_loss_df = pd.DataFrame({"loss": loss})
            repertoire_loss_df['repertoire'] = repertoire
            repertoire_loss_df['ntasks'] = repertoire.count('_') - 1 #TODO: fix this temp solution
            repertoire_loss_df['seed'] = seed
            repertoire_loss_df['epoch'] = range(len(loss))
            repertoire_loss_df['rel_loss'] = repertoire_loss_df['loss']/loss[0]
            loss_df = loss_df.append(repertoire_loss_df, ignore_index=True)
        
    return loss_df

def graph_loss(seeds:List[int], sim_set:str, repertoires:List[str], perturbation:str=None, pert_param:float=None, pert_set:str = None, smooth:bool=False, smoothing_interval:int = 5, rel_loss:bool = False, measure:str = 'loss', ax:plt.Axes = None, xlim = None, colors = None):
    """ 
    Graph loss over training.
    
    Parameters
    ----------
    seeds: random seeds for simulations
    sim_set: simulation set
    repertoires: movement repertoires
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    smooth: whether to smooth loss
    smoothing_interval: number of epochs to smooth over
    rel_loss: whether to graph relative loss
    measure: loss or test_error
    ax: axis to graph on
    xlim: x-axis limits
    colors: colors for each repertoire
    """
    
    # get loss dataframe
    loss_df = get_loss_df(seeds, sim_set, repertoires, perturbation, pert_param, pert_set,\
         smooth, smoothing_interval, measure)
    loss = 'rel_loss' if rel_loss else 'loss'

    #plot loss
    if ax is None:
        ax = plt.gca()

    colors = colors or [cm.get_cmap('plasma_r')(i) for i in np.linspace(0.1,0.9,len(repertoires))]
    g = sns.lineplot(ax = ax, data=loss_df, x="epoch", y=loss, hue = "repertoire", palette=colors)
    g.set(xlabel='Training trial', ylabel=('Relative loss' if rel_loss else 'Loss'))
    g.get_legend().remove()
    seed = seeds[0] if len(seeds) == 1 else None
    
    if xlim is not None:
        g.set_xlim(xlim)

    # save plots
    params = '_'.join(filter(None, [sim_set, str(seed) if seed else None, perturbation, pert_set,
        str(pert_param) if pert_param else None, "smooth" if smooth else None, str(xlim) if xlim else None]))
    return g, params

def graph_loss_broken_axis(seeds:List[int], sim_set:str, repertoires:List[str], perturbation:str=None, pert_param:float=None, pert_set:str = None, smooth:bool=False, rel_loss:bool = False, measure:str = 'loss', fig_ax = None):
    """
    Graph loss over training with broken axis.

    Parameters
    ----------
    seeds: random seeds for simulations
    sim_set: simulation set
    repertoires: movement repertoires
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    smooth: whether to smooth loss
    rel_loss: whether to graph relative loss
    measure: loss or test_error
    fig_ax: figure and axis to graph on

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if fig_ax is not None:
        fig, ax_left = fig_ax 
    else:
        fig = plt.figure(figsize = (3,3))
        ax_left = plt.gca()

    # ax = axes[0]
    divider = make_axes_locatable(ax_left)
    ax_right = divider.new_horizontal(size="25%", pad=0.1)
    fig.add_axes(ax_right)

    g, _ = graph_loss(seeds, sim_set, repertoires, perturbation, pert_param, pert_set, 
                    smooth=smooth, rel_loss = rel_loss, ax = ax_left, xlim = [0,50], measure = measure)
    g, _ = graph_loss(seeds, sim_set, repertoires, perturbation, pert_param, pert_set, 
                    smooth=smooth, rel_loss = rel_loss, ax = ax_right, xlim = [90,100], measure = measure)

    #break axis
    sns.despine(ax=ax_left)
    sns.despine(ax=ax_right)
    ax_right.spines['left'].set_visible(False)
    ax_right.axes.yaxis.set_visible(False)
    ax_left.set_xlabel('')
    ax_right.set_xlabel('')
    ax_left.set_xticks([0,20,40])
    ax_right.set_xticks([100])

    ax = ax_left
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    ax2 = ax_right
    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d*4, +d*4), (-d, +d), **kwargs) 
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Training trial")
    plt.ylabel("")

    params = '_'.join(filter(None, [sim_set, perturbation, pert_set,\
        str(pert_param) if pert_param else None, "smooth" if smooth else None]))
    return ax_left, ax_right, params


def get_testing_MSE(seed:int, sim_set:str, repertoire:str,  perturbation:str = None, pert_param:float= None, pert_set:str = None, noise:float= None, datafile:str = None, epoch:int= None, rotate_angle:float=0.0):
    """
    Get testing MSE for a given model.

    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    noise: noise level; if None, use training noise level
    datafile: datafile to test on
    epoch: epoch during training to test on
    rotate_angle: angle to rotate output by

    Returns
    -------
    mse: mean squared error between target and output

    """
    #get directories and model output
    outdir = get_outdir(seed, sim_set, repertoire, perturbation, pert_param, pert_set)
    datadir, _, output, _ = test_model(outdir, noise = noise, datafile = datafile, epoch = epoch)

    angle_rad = math.radians(rotate_angle)
    rotation = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],[math.sin(angle_rad), math.cos(angle_rad)]])
    output = output @ rotation

    #get data
    datname = Constants.PROJ_DIR + datadir
    test_data = np.load(datname+'.npy',allow_pickle = True).item()['test_set1']
    target = test_data['target']

    mse = ((target[:,50:,:] - output[:,50:,:])**2).mean(axis = 1)
    mse = mse.mean()

    return mse

def get_testing_MSE_by_target(seed:int, sim_set:str, repertoire:str, perturbation:str = None, pert_param:float= None, pert_set:str = None, noise:float= None, datafile:str = None, epoch:int = None, rotate_angle:float =0.0):
    """
    Get testing MSE for a given model, broken down by target.

    Parameters
    ----------
    seed: random seed for simulation
    sim_set: simulation set
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    noise: noise level; if None, use training noise level
    datafile: datafile to test on
    epoch: epoch during training to test on
    rotate_angle: angle to rotate output by
    
    Returns
    -------
    mses: list of tuples (target, mean squared error between target and output)
    """
    #get directories and model output
    outdir = get_outdir(seed, sim_set, repertoire, perturbation, pert_param, pert_set)
    datadir, _, output, _ = test_model(outdir, noise = noise, datafile = datafile, epoch=epoch)

    angle_rad = math.radians(rotate_angle)
    rotation = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],[math.sin(angle_rad), math.cos(angle_rad)]])
    output = output @ rotation

    #get data
    datname = Constants.PROJ_DIR + datadir
    test_data = np.load(datname+'.npy',allow_pickle = True).item()['test_set1']
    target = test_data['target']
    target_param = test_data['target_param']

    mses =[]
    for t in np.unique(target_param):
        idx = np.where(target_param == t)
        mse = ((target[idx,50:,:] - output[idx,50:,:])**2).mean(axis = 1)
        mse = mse.mean()
        mses.append((t, mse))

    return mses

def get_testing_MSE_df(seeds:List[int], sim_set:str, repertoires:List[str], 
    perturbation:str = None, pert_param:float = None, pert_set:str = None, noise:float = None, datafile:str = None, learning_datafile:bool = False, by_target = False, epoch = None, rotate_angle = 0.0):
    """
    Get testing MSE for several models.

    Parameters
    ----------
    seeds: list of random seeds for simulation
    sim_set: simulation set
    repertoires: list of movement repertoires
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    noise: noise level; if None, use training noise level
    datafile: datafile to test on
    learning_datafile: whether to use the datafile used for training
    by_target: whether to return MSE separated by target
    epoch: epoch during training to test on
    rotate_angle: angle to rotate output by
    """

    rows_list = []
    
    for seed in seeds:
        for repertoire in repertoires:

            if learning_datafile:
                datafile = repertoire + Constants.SIM_SET_DATA[sim_set]

            if by_target:
                mses = get_testing_MSE_by_target(seed,sim_set, repertoire,  
                                perturbation, pert_param, pert_set, 
                                noise=noise,datafile=datafile, epoch=epoch, 
                                rotate_angle=rotate_angle)
                for target_param, mse in mses:
                    dict_temp = {'seed':seed,
                        'perturbation': perturbation, 
                        'pert_param': pert_param,
                        'pert_set': pert_set,
                        'repertoire':repertoire,  
                        'target_param': target_param,                                         
                        'mse': mse,
                        'noise': noise,
                        'datafile': datafile,
                    }
                    rows_list.append(dict_temp)
            else:
                mse = get_testing_MSE(seed,sim_set, repertoire, 
                                    perturbation, pert_param, pert_set, 
                                    noise=noise,datafile=datafile, epoch=epoch,
                                    rotate_angle=rotate_angle)

                dict_temp = {'seed':seed,
                            'perturbation': perturbation, 
                            'pert_param': pert_param,
                            'pert_set': pert_set,
                            'repertoire':repertoire,                                           
                            'mse': mse,
                            'noise': noise,
                            'datafile': datafile,
                    }
                rows_list.append(dict_temp)

    mse_df = pd.DataFrame(rows_list)
    return mse_df

# GET AND PREPROCESS PYALDATA ###############################
def model_to_pyaldata (sim_set:str, seed:int, repertoire:str, perturbation:str=None, pert_param:float=0, pert_set:str = None, epoch:int = None, datafile:str = None, noise:float = None, calculate_kinematics:bool = False):
    """
    Converts model results to Pyaldata format.

    Parameters
    ----------
    sim_set: simulation set
    seed: random seed for simulation
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    epoch: epoch during training
    datafile: datafile to test model on
    noise: noise level; if None, uses noise level from training
    calculate_kinematics: whether to calculate kinematics

    Returns
    -------
    df: Pandas dataframe
        in pyaldata format

    """
    #get directories and model output
    outdir = get_outdir(seed, sim_set, repertoire, perturbation, pert_param, pert_set)
    datadir, _, output, activity1 = test_model(outdir, epoch, datafile, noise = noise)

    #get data
    datname = Constants.PROJ_DIR + datadir
    data = np.load(datname+'.npy',allow_pickle = True).item()
    test_data = data['test_set1']
    params = data['params']
    moveset = get_moveset(repertoire)

    # columns needed for pyaldata
    column_names = ['seed', 'repertoire', 'target_param', 'trial_id',
            'bin_size', 'perturbation', 'perturbation_info',
            'idx_trial_start', 'idx_target_on', 'idx_go_cue', 'idx_movement_on',
            'idx_trial_end', 'pos']
    df = pd.DataFrame(columns = column_names)

    ntrials = len(test_data['idxoutofall'])
    tsteps = params['tsteps']
    dt = params['dt']

    #populate columns
    df['trial_id'] = test_data['idxoutofall']
    df['seed'] = seed
    df['repertoire'] = repertoire
    df['target_param'] = test_data['target_param']
    df['stimulus_param'] = test_data['stimulus_param'] 
    df['bin_size'] = params['dt']
    df['perturbation'] = perturbation
    df['perturbation_info'] = pert_param
    df['idx_trial_start'] = 0
    df['idx_target_on'] = test_data['cue_onset']
    df['idx_go_cue'] = test_data['go_onset']
    df['idx_trial_end'] = tsteps-1
    df['pos'] = [output[i,:] for i in range(ntrials)] 
    df['MCx_rates'] =[activity1[i,:] for i in range(ntrials)] 
    df['training_epoch'] = epoch

    if calculate_kinematics:
        #calculate vel, accel, and speed
        vels = np.zeros((ntrials, tsteps, 2))
        pos = output
        for trial in range(ntrials):
            go_step = df['idx_go_cue'][trial]
            for tstep in range(go_step, tsteps):
                #calculate velocity
                vels[trial, tstep, 0] = (pos[trial,tstep,0]-pos[trial,tstep-1,0])/dt
                vels[trial, tstep, 1] = (pos[trial,tstep,1]-pos[trial,tstep-1,1])/dt

        df['vel'] = [vels[i] for i in range(ntrials)]

    return df

def preprocess_data(pyal_df:pd.DataFrame, epoch_fun:Callable = None, subtract_mean:bool = False):
    """
    Preprocesses pyaldata by merging and smoothing signals, and restricting time interval

    Parameters
    ----------
    pyal_df: pyaldata dataframe
    epoch_fun: function to restrict time interval
    subtract_mean: whether to subtract cross-condition mean

    Returns
    -------
    df: Pandas dataframe
        in pyaldata format
    """
    if 'M1_rates' in pyal_df.columns: 
        pyal_df = pyal.merge_signals(pyal_df, ["M1_rates", "PMd_rates"], "MCx_rates")
        pyal_df = pyal.smooth_signals(pyal_df, ["M1_rates", "PMd_rates", "MCx_rates"])
    else:
        pyal_df = pyal.smooth_signals(pyal_df, ["MCx_rates"])
    
    pyal_df = pyal.add_movement_onset(pyal_df, method = 'peaks')
    if epoch_fun is not None:
        pyal_df = pyal.restrict_to_interval(pyal_df, epoch_fun = epoch_fun)

    if subtract_mean: 
        pyal_df = pyal.subtract_cross_condition_mean(pyal_df)
        
    return pyal_df

def get_pyaldata(sim_set:str, seed:int, repertoire:str, perturbation:str=None, pert_param:float=None, pert_set:str=None, epoch_fun:Callable = None, epoch:int = None, datafile:str = None, subtract_mean:bool=False, noise:float = None, calculate_kinematics:bool = False):
    """
    Converts model results to Pyaldata format and preprocess data.

    Parameters
    ----------
    sim_set: simulation set
    seed: random seed for simulation
    repertoire: movement repertoire
    perturbation: perturbation type ('rotation', 'reassoc')
    pert_param: perturbation parameter
    pert_set: perturbation set
    epoch_fun: function to restrict time interval
    epoch: epoch during training
    datafile: datafile to test model on
    subtract_mean: whether to subtract cross-condition mean
    noise: noise level; if None, uses noise level from training
    calculate_kinematics: whether to calculate kinematics

    Returns
    -------
    df: Pandas dataframe
        in pyaldata format
    """

    pyal_df = model_to_pyaldata(sim_set, seed, repertoire, perturbation, pert_param, pert_set, 
                                epoch = epoch, datafile = datafile, noise = noise, calculate_kinematics = calculate_kinematics)
    pyal_df = preprocess_data(pyal_df, epoch_fun = epoch_fun, subtract_mean=subtract_mean)

    return pyal_df

def perform_pca (pyal_df:pd.DataFrame, pca_dims:int):
    """
    Performs PCA on pyaldata
    
    Parameters
    ----------
    pyal_df: pyaldata dataframe
    pca_dims: number of PCA dimensions

    Returns
    -------
    df: Pandas dataframe
        in pyaldata format
    """
    if 'M1_rates' in pyal_df.columns:
        pyal_df = pyal.dim_reduce(pyal_df, PCA(pca_dims), "M1_rates", "M1_pca")
        pyal_df = pyal.dim_reduce(pyal_df, PCA(pca_dims), "PMd_rates", "PMd_pca")
    pyal_df = pyal.dim_reduce(pyal_df, PCA(pca_dims), "MCx_rates", "both_pca")
    
    return pyal_df


# plotting comparisons ##########################
def compare_measure(measures_df:pd.DataFrame, measure:str, reps:List[str], norm_factor:str= None, estimator:Callable = np.median, stats_alternative:str = 'greater', ci:int = 95, seeds = Constants.SEEDS):
    """
    Plots pointplot across seeds for each repertoire. Values for each seed are connected by lines for each repertoire. Prints stats for each comparison.

    Parameters
    ----------
    measures_df: dataframe with measures
    measure: measure to compare
    reps: list of repertoires to compare
    norm_factor: factor to normalize measure by
    estimator: estimator for pointplot
    stats_alternative: "less", "greater", "two-sided"
    ci: confidence interval for stats
    seeds: list of seeds to plot

    """
    if norm_factor is not None:
        measures_df[measure+'_norm'] = measures_df[measure] * measures_df[norm_factor] 
        df = measures_df[['seed','repertoire',measure, measure+'_norm']]       
        df = df.explode(measure+'_norm')
        measure = measure+'_norm'
    else:
        df = measures_df[['seed','repertoire',measure]]        
        df = df.explode(measure)
    g= pointplot_across_seeds(df, reps, seeds, measure, estimator = estimator, ci = ci)

    #stats
    measures_df[measure + '_est'] = measures_df.apply(lambda row: estimator(row[measure]), axis = 1)
    for i, rep1 in enumerate(reps[:-1]):
        a = measures_df[(measures_df.repertoire == reps[i])][measure + '_est'].values
        b = measures_df[(measures_df.repertoire == reps[i+1])][measure + '_est'].values

        #check normality: this is not normal
        # print(abr_dict[1], scipy.stats.shapiro(a_measure))
        _, pnorm = scipy.stats.wilcoxon(a, b, alternative = stats_alternative)
        text = get_signif_annot(pnorm)
        print(pnorm)

        g.annotate(text, xy=((i+1)*0.25, 0.9), xytext=((i+1)*0.25, 0.95), xycoords='axes fraction', fontsize = 'x-large',ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, widthB=1.3, lengthB=0.4', lw=1))
    return g

def pointplot_across_seeds (data:pd.DataFrame, repertoires:List[str], seeds:List[int], y:str, estimator=np.median, ax:plt.Axes = None, ci:int = 95):
    """
    Plots pointplot across seeds for each repertoire. Values for each seed are connected by lines for each repertoire.

    Parameters
    ----------
    data: dataframe with model results
    repertoires: list of repertoires
    seeds: list of seeds
    y: measure to plot
    estimator: estimator for pointplot
    ax: axis to plot on
    ci: confidence interval
    """

    moveset = get_moveset(repertoires[0])
    abr_dict = Constants.REPERTOIRE_ABR_DICTS[moveset]
    df = data[(data.repertoire.isin(repertoires)) & (data.seed.isin(seeds))]

    if ax is None:
        if len(repertoires) <=2:
            plt.figure(figsize=(2.5,3))
        else:
            plt.figure(figsize=(3,4))
        ax = plt.gca()

    #plot connecting lines
    line_colors = Constants.SEED_COLORS[:len(seeds)]
    g = sns.pointplot(x="repertoire", y=y, data=df, markers="", estimator = estimator,
                    join=True, ci=None, hue = 'seed', palette = line_colors, ax = ax)
    plt.setp(g.lines, zorder=0)
    plt.setp(g.collections, zorder=0, label="")

    #plot points
    point_colors = Constants.REPERTOIRE_COLORS
    colors_dict = dict(zip(repertoires, point_colors[:len(repertoires)]))
    for seed in seeds:
        df_ = df[df.seed == seed]
        g = sns.pointplot(x="repertoire", y=y, data=df_, estimator = estimator,
                        palette=colors_dict, hue = 'repertoire', join = False,
                        ci=ci, ax = ax)

    g.set_xticklabels([abr_dict[s] if s in abr_dict.keys() else s for s in repertoires])
    g.set_xlabel('Repertoire')
    ymin, ymax = g.get_ylim()
    g.set_ylim([None, ymax+(ymax-ymin)*0.2])
    plt.xticks(rotation = 45)

    plt.legend([],[], frameon=False)
    return g


    
# def tangling(X):
#     """
#     Calculates tangling measure from Churchland 2018

#     Parameters
#     ----------
#     X : array
#         c conditions x n features x t timesteps data
   
#     Returns
#     Q: array of length t-1
#         Q(t) = max [(||dX_t - dX_t'|| ^ 2)/ (||X_t - X_t'|| ^ 2 + epsilon)
#     -------

#     """
#     from sklearn.metrics import pairwise_distances

#     dX = np.concatenate(np.diff(X,axis = 2),axis = 1)
#     X_concat = np.concatenate(X[:,:,1:], axis = 1)
    
#     epsilon = 0.1*np.mean(np.square(X_concat))

#     X_concat_dist = pairwise_distances(X_concat.T, metric = 'sqeuclidean')
#     dX_dist = pairwise_distances(dX.T, metric = 'sqeuclidean')

#     # calculate tangling for all timesteps
#     Q_t = dX_dist / (X_concat_dist + epsilon)
#     Q = np.max(Q_t, axis = 1)
    
#     return Q


def plot_3d(df:pd.DataFrame, signal:str, dims:int, rel_start:int,elev:int = None, azim:int = None, ax:plt.Axes = None, alpha:float = 1, linestyle:str = 'solid'):
    """
    Plots 3d trajectories for a given signal

    Parameters
    ----------
    df: pyaldata dataframe
    signal: signal to plot
    dims: dimensions to plot
    rel_start: relative start time
    elev: elevation for 3d plot
    azim: azimuth for 3d plot
    ax: axis to plot on
    alpha: opacity
    linestyle: linestyle
    """

    if len(np.unique(df.target_param.values)) == 1:
        colormap = get_colormap(df.target_param.values, cmap = Constants.MOVEMENT_CMAP)
    else:
        colormap = get_colormap(df.target_param.values, cmap = 'viridis_r')

    orig_targets = df.target_param.values
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    #plot for each stimulus
    for i, df_stim in enumerate(df[signal]):
        #plot trajectories
        ax.plot(df_stim[:, dims[0]-1].T, 
                df_stim[:, dims[1]-1].T, 
                df_stim[:, dims[2]-1].T,
                linewidth = 3,
                linestyle = linestyle,
                color = colormap[orig_targets[i]],
                alpha = alpha,
                )
        #plot start
        if -rel_start != 0:
            ax.scatter(df_stim[0, dims[0]-1].T, 
                       df_stim[0, dims[1]-1].T, 
                       df_stim[0, dims[2]-1].T,
                       alpha = alpha, 
                       color = colormap[orig_targets[i]],
                       s= 140, 
                       marker = '.'
                       )
        #plot end
        if -rel_start != (df_stim.shape[0]-1):
            ax.scatter(df_stim[-1, dims[0]-1].T, 
                       df_stim[-1, dims[1]-1].T,
                       df_stim[-1, dims[2]-1].T,
                       color = colormap[orig_targets[i]],
                       s= 50,
                       alpha = alpha, 
                       marker = 'D'
                       )
        #plot ref point
        ax.scatter(df_stim[-rel_start, dims[0]-1].T, 
                   df_stim[-rel_start, dims[1]-1].T,
                   df_stim[-rel_start, dims[2]-1].T,
                   color = colormap[orig_targets[i]],
                   s= 250,
                   alpha = alpha,
                   marker = '+'
                   )

    ax.set_xlabel('PC'+str(dims[0]))
    ax.set_ylabel('PC'+str(dims[1]))
    ax.set_zlabel('PC'+str(dims[2]))
#     ax.grid(False)
    
    if elev is not None:
        ax.elev = elev
    if azim is not None:
        ax.azim = azim
        
    plt.tight_layout()
    return ax

def pointplot_across_repertoires(data: pd.DataFrame, repertoires: List[str], y:str, color = 'grey', linestyles:List[str] = ['solid'], ax:plt.Axes = None, abr_labels:bool = True):
    """
    Plots a pointplot across repertoires. Points and bars, means and 95% confidence intervals. Values for different seeds shown as opaque points. 

    Parameters
    ----------
    data: dataframe 
    repertoires: list of repertoires to plot
    y: measure to plot
    color: color of the connecting line
    linestyles: list of linestyles for the connecting line
    ax: axis to plot on
    abr_labels: whether to use abbreviated labels for the repertoires
    """

    df = data[data.repertoire.isin(repertoires)]
    colors = Constants.REPERTOIRE_COLORS[-len(repertoires):] #TODO: add repertoire colors dict in constants

    #make one plot for the line without points and errorbars
    g = sns.pointplot(x="repertoire", y=y, data=df, markers="", join=True, ci=None, color=color, linestyles = linestyles, ax = ax)
    #make one plot for the points without the connecting line
    g = sns.pointplot(x="repertoire", y=y, data=df,
                    palette=colors, ax = ax)
    if abr_labels:
        moveset = get_moveset(repertoires[0])
        abr_dict = Constants.REPERTOIRE_ABR_DICTS[moveset]
        g.set_xticklabels([abr_dict[s] for s in repertoires])
    g.set_xlabel('Repertoire')

    return g

# WEIGHTS #####################################################################
def rel_dw(dw:np.array, init_weights:np.array):
    """ 
    Calculate relative change in weights.
    
    Parameters
    ----------
    dw: change in weights
    init_weights: initial weights
    """
    return np.nan_to_num(np.divide(np.abs(dw), init_weights))

def get_model_weights(outdir:str, before_training:bool = False, epoch:int = None):
    """ 
    Get model weights before/after training or at a training epoch.
    
    Parameters
    ----------
    outdir: directory where model weights are saved
    before_training: whether to get weights before training
    epoch: epoch to get weights for, if None, get weights after training

    Returns
    -------
    weights_dict: dictionary of weights with "input", "readout", and "rec" keys
    """

    #get model weights for specific epoch during training
    if (epoch is not None) & (not before_training):
        epoch_str = '_'+str(epoch)
        temp = torch.load(outdir+'model'+ epoch_str)['model_state_dict']
        
        in_w = temp["rnn_l1.weight_ih_l0"].cpu().numpy()
        out_w = temp["noarm_output.weight"].cpu().numpy()
        rec_w = temp["rnn_l1.weight_hh_l0"].cpu().numpy()
    else:
        train_data = get_train_data(outdir)
        
        #get model weights before training
        if before_training:
            weights = train_data['params0']
        #get model weights after training
        else:
            weights = train_data['params1']
        
        in_w = weights['wihl1']
        out_w = weights['wout']
        rec_w = weights['whhl1']

    weights_dict = {
        "input" : in_w,
        "readout": out_w,
        "rec": rec_w,
    }

    return weights_dict

def get_weight_changes(outdir:str, epoch:int = None):
    """ 
    Get weight changes during training.
    
    Parameters
    ----------
    outdir: directory where model weights are saved
    epoch: epoch to get weights for, if None, get weights after training
    
    Returns
    -------
    weight_changes: dictionary of weight changes with "input", "readout", and "rec" keys
    """

    weights_before = get_model_weights(outdir, before_training=True, epoch = epoch)
    weights_after = get_model_weights(outdir, before_training=False, epoch = epoch)

    weight_changes = {}
    for key in weights_before.keys():
        weight_changes[key] = weights_after[key] - weights_before[key]
        
    return weight_changes

# EXPERIMENTAL DATA #####################################################################

def canoncorr(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`

    Parameters
    ----------
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    Returns
    -------
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    from scipy.linalg import qr, svd, inv
    import logging

    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0);
    Y = Y - np.mean(Y,0);

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]));

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]));

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY];
        T22 = T22[:rankY,:rankY];

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY);
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1);
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1);
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V

def get_target_id(trial):
    """
    Get target id from trial info for experimental data.

    Parameters
    ----------
    trial: trial info (row of dataframe)

    Returns
    -------
    target_id: target id, integer from 1 to 8

    """
    direction = (trial.target_direction + 2 * np.pi) if trial.target_direction < 0 else trial.target_direction
    return int(np.round((direction) / (0.25*np.pi))) + 1

def preprocess_exp_data(exp_df:pd.DataFrame, repertoire:str, subtract_mean:bool = True, remove_low_firing:bool = True):
    """
    Preprocess experimental data for analysis.

    Parameters
    ----------
    exp_df: dataframe of experimental data
    repertoire: movement repertoire
    subtract_mean: whether to subtract mean from firing rates
    remove_low_firing: whether to remove low firing neurons

    Returns
    -------
    exp_df: preprocessed dataframe, pyaldata format

    """

    #get and preprocess data
    exp_df = pyal.combine_time_bins(exp_df,int(Constants.EXP_BIN_SIZE/Constants.BIN_SIZE))
    if remove_low_firing:
        exp_df = pyal.remove_low_firing_neurons(exp_df, "M1_spikes",  5)
        exp_df = pyal.remove_low_firing_neurons(exp_df, "PMd_spikes", 5)
    exp_df = pyal.transform_signal(exp_df, "M1_spikes",  'sqrt')
    exp_df = pyal.transform_signal(exp_df, "PMd_spikes", 'sqrt')
    exp_df = pyal.merge_signals(exp_df, ["M1_spikes", "PMd_spikes"], "MCx_spikes")
    exp_df = pyal.add_firing_rates(exp_df, 'smooth')
    exp_df = pyal.select_trials(exp_df, "result == 'R'")

    prep_exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue',
                                    rel_start=int(Constants.WINDOW_prep_exec[0]/Constants.EXP_BIN_SIZE),
                                    rel_end=int(Constants.WINDOW_prep_exec[1]/Constants.EXP_BIN_SIZE)
                                    )

    exp_df = pyal.restrict_to_interval(exp_df, epoch_fun = prep_exec_epoch)
    if subtract_mean:
        exp_df = pyal.subtract_cross_condition_mean(exp_df)

    # add parameters so the dataframe is the same as simulations
    exp_df["target_id"] = exp_df.apply(get_target_id, axis=1)
    exp_df["target_direction_orig"] = exp_df['target_id']
    exp_df["task"] = 1
    exp_df["repertoire"] = repertoire
    exp_df["perturbation_info"] = 0
    exp_df['target_param'] = exp_df['target_direction']
    return exp_df

# SUBSPACE MEASURES #####################################################################

def manifold_overlap (data1:np.array, data2:np.array, pca_dims:int = None, pc1:np.array = None):
    """
    Find manifold overlap between two subspaces
    Based on measure in Feulner 2021, adapted from Elsayed 2016

    Parameters
    ----------
    data1 : n_samples x n_features data
    data2 : n_samples x n_features data
    pca_dims: dimensions to use for PCA
    pc1: n x q matrix that defines q-dimensional plane in an n-dim Euclidean space
        principal components calculated for data1
        (e.g. matrix with q leading principal components as column vectors)

    Returns
    -------
    overlap: float
        overlap between subspaces

    """
    assert((pca_dims is not None)|(pc1 is not None))

    cov1 = np.cov(data1.T)
    cov2 = np.cov(data2.T)

    if pc1 is None:
        pc1 = principal_components(data1, pca_dims).T
        
    proj_original = pc1.T@cov1@pc1
    overlap_original = np.trace(proj_original)/ np.trace(cov1)
    proj_pert = pc1.T@cov2@pc1
    overlap_pert = np.trace(proj_pert)/ np.trace(cov2)
    overlap = overlap_pert/overlap_original
    return overlap

def manifold_overlap_control (data1:np.array, pca_dims:int = Constants.PCA_DIMS):
    """
    Find lower-bound (control) for manifold overlap between two subspaces
    Based on measure in Feulner 2021, adapted from Elsayed 2016

    Parameters
    ----------
    data1 : n_samples x n_features data
    pca_dims: idimensions to use for PCA

    Returns
    -------
    overlap: float
    """
    rng = np.random.default_rng(12345)

    cov1 = np.cov(data1.T)
    pc1 = principal_components(data1, pca_dims).T
        
    proj_original = pc1.T@cov1@pc1
    overlap_original = np.trace(proj_original)/ np.trace(cov1)
    proj_pert = pc1.T@rng.permutation(cov1)@pc1
    overlap_pert = np.trace(proj_pert)/ np.trace(cov1)
    overlap = overlap_pert/overlap_original
    return overlap

#FROM BENEURO
def variance_in_subspace(X, W):
    """
    Variance in a given subspace

    Parameters
    ----------
    X : 2D np.ndarray
        n_samples x n_features data array
    W : 2D np.ndarray
        n_features x n_components projection matrix
        
    Returns
    -------
    variance in the subspace
    """
    W_norm = W / np.linalg.norm(W, axis=0)
    return np.trace(W_norm.T @ np.cov(X.T) @ W_norm)

# def svd_pr(arr):
#     """
#     Calculate participation ratio using SVD

#     Parameters
#     ----------
#     arr: array
#         n_samples x n_features data
   
#     Returns
#     -------
#     estimated dimensionality

#     """
#     s = scipy.linalg.svdvals(arr)
    
#     return np.sum(s) ** 2 / np.sum(s ** 2)

def principal_components(data:np.array, pca_dims:int):
    """
    Calculate principal components using PCA

    Parameters
    ----------
    data: n_samples x n_features data
    pca_dims: number of dimensions to use for PCA
   
    Returns
    -------
    principal components: array 
        n x q matrix that defines q-dimensional plane in an n-dim Euclidean space

    """
    pca_dims = min(data.shape[0], data.shape[1], pca_dims)
    model = PCA(pca_dims).fit(data)
    return model.components_