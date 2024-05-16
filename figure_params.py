import matplotlib as mpl
from constants import Constants
import os

def set_rc_params(dictArg:dict ={}):
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['figure.dpi'] = 96
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams["axes.formatter.limits"] = (-3,5)
    mpl.rcParams['figure.figsize'] = (4,3)

    for key,val in dictArg.items():
        mpl.rcParams[key] = val

def setup_directories():
    if not os.path.exists(Constants.FIG_FOLDER):
        os.makedirs(Constants.FIG_FOLDER)
    if not os.path.exists(Constants.PROCESSED_DATA_FOLDER):
        os.makedirs(Constants.PROCESSED_DATA_FOLDER)
    for key in Constants.SIM_SET_DATA.keys():
        if not os.path.exists(Constants.PROCESSED_DATA_FOLDER+key):
            os.makedirs(Constants.PROCESSED_DATA_FOLDER+key)
    if not os.path.exists(Constants.PROCESSED_DATA_FOLDER+'exp'):
        os.makedirs(Constants.PROCESSED_DATA_FOLDER+'exp')
        
