#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Author: Barbara Feulner, adapted by Joanna Chang
Create pytorch dataset from experimental data.

Code shows how dataset_uni.npy and dataset_centerout.npy were created from experimental data.

Create dataset (dict) for movement training.
The output dimensions are uncorrelated in this implementation.
The output has time components.
The target and the go cue vary from trial to trial.

Structure (keys):
---------
    - target_id: ntrials x 1
    - target_output: ntrials x time x noutputdimensions
    - params
    - stimulus ntrials x time x ninput dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (4,3)
mpl.rcParams['figure.dpi'] = 96
import pandas as pd
import scipy.signal as sg
from scipy.ndimage import convolve1d
import constants
import math
# params #################################
uni_transformation = True
use_velocities = False
savname = 'dataset' + ('_vel' if use_velocities else '') + ('_uni' if uni_transformation else '_centerout')
savdir = constants.Constants.PROJ_DIR +constants.Constants.DATA_FOLDER
figdir = 'results/'

ntargets = 8 # how many different targets
tsteps = 2000 # how many time steps in one trial
dt = 0.01

output_dim = 2
input_dim = 3
input_max = 2

train_trials = 2016 #divisible by ntargets #None

np.random.seed(0)
######################################### load data and create summary
# import scipy.io as spio
# import scipy.signal as sg
# from scipy.ndimage import convolve1d
# import os

# tab = pd.read_excel(constants.Constants.PROJ_DIR + constants.Constants.EXP_DATA_FOLDER + 'summary.xlsx',engine='openpyxl')
# datadir = constants.Constants.PROJ_DIR + constants.Constants.EXP_DATA_FOLDER
# target_output = [] # just add all trials
# exp_info = [] # create big pandas table with info about trials in target_output
# exp_info_cols = ['Monkey','ExpType','Date','Epoch','Go','Shift','Target',
#                  'Success','Pinf1','Pinf2','dt','x0','y0','r','TrialCount','TabL']
# for l in range(tab.shape[0]):
#    monkey = tab.iloc[l].Monkey
#    exptype = tab.iloc[l].ExpType
#    date = tab.iloc[l].Date   
#    loadname = datadir + monkey + '/' + tab.iloc[l].Filename
#    data = spio.loadmat(loadname+'.mat',squeeze_me=True)['trial_data']
   
#    # params
#    ntrials = data.size
#    dt = data[0]['bin_size']
   
#    x0 = []
#    y0 = []
#    r = []
#    for trial_idx in range(ntrials):
#        start = data[trial_idx]['idx_target_on']
#        end = data[trial_idx]['idx_trial_end']
#        pos = data[trial_idx]['pos'][start,:]
#        pos2 = data[trial_idx]['pos'][end,:]
#        x0.append(pos[0])
#        y0.append(pos[1])
#        r.append(np.linalg.norm(pos2-pos))
#    x0 = np.mean(x0) # in cm
#    y0 = np.mean(y0) # in cm
#    r = np.mean(r) # in cm
   
#    trial_count = 0
#    for trial_idx in range(ntrials):
#        start = data[trial_idx]['idx_target_on']
#        end = data[trial_idx]['idx_trial_end']
#        len = end-start
#        # trial info
#        target = data[trial_idx]['target_direction']
#        result = data[trial_idx]['result'] # 'R' successful, 'I' failed 
#        epoch = data[trial_idx]['epoch'] # 'BL' AD' 'WO'
#        if epoch=='AD':
#            trial_count += 1
#        elif epoch=='WO':
#            trial_count = 0
#        # timing and trial events
#        trial_end = len
#        trial_go = data[trial_idx]['idx_go_cue']-start
#        shift = tsteps-len
#        go = shift + trial_go
#        # kinematics
#        pos = data[trial_idx]['pos'][start:end,:] # in cm
#        vel = data[trial_idx]['vel'][start:end,:]
#        temp = np.concatenate((pos,vel),axis=-1)
#        to_temp = np.ones((tsteps,4))
#        to_temp[:,] = temp[0,:] 
#        to_temp[shift:,:] = temp
#        target_output.append(to_temp)
#        if exptype=='VR' and epoch=='AD':
#            exp_info.append([monkey,exptype,date,epoch,go,shift,target,result,
#                            data[trial_idx]['perturbation_info'],None,dt,x0,y0,
#                            r,trial_count,l])
#        elif exptype=='FF' and epoch=='AD':
#            exp_info.append([monkey,exptype,date,epoch,go,shift,target,result,
#                            data[trial_idx]['perturbation_info'][0],
#                            data[trial_idx]['perturbation_info'][1],dt,x0,y0,r,
#                            trial_count,l])
#        else:
#            exp_info.append([monkey,exptype,date,epoch,go,shift,target,result,
#                              None,None,dt,x0,y0,r,trial_count,l])
#    print(str(l)+' done')
       
# exp_info = pd.DataFrame(exp_info,columns=exp_info_cols)

# #%% how to select the timing 
# plt.figure()
# plt.hist(exp_info.Shift.values,np.linspace(1600,2000,100))

# #%% chunk it
# startpoint = 1600
# target_output = np.array(target_output)
# target_output = target_output[:,startpoint:]
# exp_info.Go -= startpoint
# exp_info.Shift -= startpoint
# tsteps = target_output.shape[1]

# #%% save it
# exp_info.to_excel('matt_summary.xlsx')
# np.save('matt_posvel',target_output)
###################################################################
# %% load it
exp_dir = constants.Constants.PROJ_DIR +constants.Constants.EXP_DATA_FOLDER
exp_info = pd.read_excel(exp_dir + 'matt_summary.xlsx',engine='openpyxl')
target_output = np.load(exp_dir + 'matt_posvel.npy',allow_pickle=True)
# %% now, translate it to the data set format used for pytorch
#########################################

#%%
tsteps = target_output.shape[1]

# select which type of session to take
idx = (exp_info.Monkey.values=='Chewie') & (exp_info.Epoch.values=='BL') \
        & (exp_info.Success=='R')


#%%
idx = idx.values
ntrials = np.sum(idx)
print("All trials", ntrials)

#target directions and positions
phi = np.unique(exp_info.Target.values)
target_phis = exp_info[idx].Target.values
x0 = exp_info[idx].x0.values
y0 = exp_info[idx].y0.values
tout = target_output[idx] #actual positions

target_id = []
for j in range(ntrials):
    target_id.append(np.where(target_phis[j]==phi)[0][0])
target_id = np.array(target_id)

#%%
# exclude error trials
target_angles = np.linspace(-np.pi,np.pi,8,endpoint=False)

target_angles = np.roll(target_angles,-1)
temp_xy = tout.copy()
temp_xy[:,:,0] -= x0[:,None]
temp_xy[:,:,1] -= y0[:,None]
ang = np.arctan2(temp_xy[:,-1,1],temp_xy[:,-1,0])
rad = np.linalg.norm(temp_xy[:,-1,:],axis=-1)
difang = ang-target_angles[target_id]
difang[difang>np.pi] = difang[difang>np.pi]-np.pi*2
difang[difang<-np.pi] = difang[difang<-np.pi]+np.pi*2
difang = np.abs(difang)
fail_trials = difang>0.5
idx[idx] = idx[idx] & (fail_trials==False) & (rad>7) & (rad<10) 
idx[480] = False #strange trajectory at one time step

# reassign data (now with failed trials excluded)
ntrials = np.sum(idx)
print("Successful trials", ntrials)
stimulus = np.zeros((ntrials,tsteps,input_dim))
phi = np.unique(exp_info.Target.values)
cue_on = exp_info[idx].Shift.values
go = exp_info[idx].Go.values
target_phis = exp_info[idx].Target.values
x0 = exp_info[idx].x0.values
y0 = exp_info[idx].y0.values
session_id = exp_info[idx].TabL.values
tout = target_output[idx]

#set to workplace center
tout[:,:,0] = (tout[:,:,0] - x0[:,None]) 
tout[:,:,1] = (tout[:,:,1] - y0[:,None]) 

target_id = []
for j in range(ntrials):
    target_id.append(np.where(target_phis[j]==phi)[0][0])

    #stimulus: (hold, 2cos, 2sin)
    stimulus[j,0:go[j],0] = input_max 
    stimulus[j,cue_on[j]:,1] = input_max*np.cos(target_phis[j])
    stimulus[j,cue_on[j]:,2] = input_max*np.sin(target_phis[j])
target_id = np.array(target_id)

#%%
if uni_transformation:
    #transform all trajectories to reach to 0 degrees
    for j in range(ntrials):
        stimulus[j,cue_on[j]:,1] = input_max*np.cos(0)
        stimulus[j,cue_on[j]:,2] = input_max*np.sin(0)
        angle = target_phis[j]
        rotation = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
        tout[j,:,:2] = tout[j,:,:2] @ rotation
        tout[j,:,-2:] = tout[j,:,-2:] @ rotation
    target_id = np.repeat(np.where(phi == 0)[0][0], ntrials)
    target_phis = np.repeat(0, ntrials)
    phi = np.unique(target_phis)

# create testset
test_trials = 192 #divisible by ntargets
assert(test_trials % len(phi) == 0)
test_idx = []
for p in phi:
    temp_idx = np.where(target_phis == p)[0]
    temp_idx = np.random.choice(temp_idx, int(test_trials/len(phi)), replace = False)
    test_idx.extend(temp_idx)

test_set1 = {'target': (tout[test_idx,:,output_dim:] if use_velocities else tout[test_idx,:,:output_dim]),
             'stimulus':stimulus[test_idx],
             'cue_onset':cue_on[test_idx],
             'go_onset':go[test_idx],
             'target_id':target_id[test_idx],
             'target_param': target_phis[test_idx],
             'idxoutofall':np.where(idx)[0][test_idx],
             }

train_idx = np.setdiff1d(range(ntrials), test_idx)

print(ntrials, len(test_idx), len(train_idx))
# save it
params = {'ntargets':ntargets,
          'tsteps':tsteps,
          'ntrials':len(train_idx),
          'output_dim':output_dim,
          'input_dim':input_dim,
          'dt':dt}

test_set1['params'] = params.copy()
test_set1['params']['ntrials'] = len(test_idx)

dic = {'params':params,
       'target':(tout[train_idx,:,output_dim:] if use_velocities else tout[train_idx,:,:output_dim]),
       'stimulus':stimulus[train_idx],
       'go_onset':go[train_idx],
       'cue_onset':cue_on[train_idx],
       'target_id':target_id[train_idx],
       'target_param': target_phis[train_idx],
       'test_set1':test_set1,
       'idxoutofall':np.where(idx)[0][train_idx],
       }


np.save(savdir+savname,dic)

# %% PLOTS (test if correct)
def calc_psth(act,tar):
    psth = []
    for i in range(8):
        psth.append(np.mean(act[tar==i],axis=0))
    psth = np.array(psth)
    return psth

temp_xy = dic['target'].copy()
cmap = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)]

#plot position
idx = 4
plt.figure()
temp_xy = temp_xy[:idx]

if use_velocities:
    positions = np.zeros(temp_xy.shape)
    for j in range(temp_xy.shape[1]):
        positions[:,j,:] = positions[:,j-1,:] + temp_xy[:,j,:]*dt
else:
    positions = temp_xy
for j in range(positions.shape[0]):
    plt.plot(positions[j,:,0].T,positions[j,:,1].T,color=cmap[target_id[j]],alpha=0.4)
plt.ylim([-5,5])

plt.figure()
positions = tout.copy()[:idx]
for j in range(positions.shape[0]):
    plt.plot(positions[j,:,0].T,positions[j,:,1].T,color=cmap[target_id[j]],alpha=0.4)
plt.ylim([-5,5])

# plt.title('%d trials'%len(target_id))
# plt.xlim([-10,10])
# plt.ylim([-10,10])
# plt.savefig(savdir+figdir+savname+'_x0y0shifted.png',bbox_inches='tight',dpi=300)

# # check reach direction
# rad = np.linalg.norm(temp_xy[:,-1,:],axis=-1)
# plt.figure()
# plt.hist(rad,100)
# plt.xlabel('Reach distance')
# plt.ylabel('Trials')
# plt.savefig(savdir+figdir+savname+'_reachdist.png',bbox_inches='tight',dpi=300)
  
# # check endpoints for each target  
# ang = np.arctan2(temp_xy[:,-1,1],temp_xy[:,-1,0])
# plt.figure()
# plt.scatter(target_id,ang,c=['k'],alpha=0.3)
# plt.xlabel('Endpoint angle')
# plt.ylabel('Trials')
# plt.savefig(savdir+figdir+savname+'_endpointangles.png',bbox_inches='tight',dpi=300)

# # check constructed stimulus
# tid = 0
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(tout[tid])
# plt.ylabel('pos/vel')
# plt.subplot(3,1,2)
# plt.plot(stimulus[tid,:,:])
# plt.ylabel('stim')
# plt.savefig(savdir+figdir+savname+'_teststim.png',bbox_inches='tight',dpi=300)

# %%
