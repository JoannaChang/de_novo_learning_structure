import numpy as np
import pyaldata as pyal
from matplotlib import cm

class Constants:
    # DIRECTORIES ##################################################
    PROJ_DIR="/home/jcc319/structure_proj/" #change based on your directory structure
    RESULTS_FOLDER = "results/"
    DATA_FOLDER = "data/"
    EXP_DATA_FOLDER = "exp_data/"
    FIG_FOLDER = PROJ_DIR + 'figures/'
    PROCESSED_DATA_FOLDER = FIG_FOLDER + 'processed/'

    # RANDOM #######################################################
    RNG = np.random.default_rng(np.random.SeedSequence(12345))
    SEEDS = list(range(1000000,1000010))
    SEED_EX = SEEDS[0]

    # DATA PROCESSING #######################
    PCA_DIMS = 10

    # FIGS/COLORS ##################################################
    FIG_FORMAT = 'pdf'
    MOVEMENT_CMAP = 'viridis'
    REPERTOIRE_CMAP = 'plasma_r'
    SEED_CMAP = 'Greys'
    REPERTOIRE_COLORS = [cm.get_cmap('plasma_r')(i) for i in np.linspace(0.1,0.9,4)]
    SEED_COLORS = [cm.get_cmap('Greys')(i) for i in np.linspace(0.2,0.9,len(SEEDS))]

    # EPOCHS #######################################################
    MOVE_ON_THRESHOLD = 9
    BIN_SIZE = .01  # sec
    EXP_BIN_SIZE = .03 # sec

    WINDOW_prep_exec = (-.50, 1.0)  # sec
    WINDOW_prep = (-.50, 0.0)  # sec
    WINDOW_exec = (0.0, 1.0)  # sec
    prep_exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue',
                                        rel_start=int(WINDOW_prep_exec[0]/BIN_SIZE),
                                        rel_end=int(WINDOW_prep_exec[1]/BIN_SIZE)
                                        )
    prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue', 
                                        rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                        rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                        )
    exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue', 
                                        rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                        rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                        )

    # TRAJECTORIES #################################################
    MAX_Y_POS = 10
    YOFFSET = 0

    # MOVESETS #####################################################
    ## CO_MOVESET for centerout tasks with 8 targets (like in experiments)
    CO_MOVESET = 'co'
    COS = ['centerout']
    COS_ABR = ['CO']

    ## UNI_MOVESET for tasks with 1-4 targets
    UNI_MOVESET = 'uni'
    UNIS = [
        'uni_10.0',
        'uni_2movs_10_50',
        'uni_3movs_10_50',
        'uni_4movs_10_50',
        ]

    UNIS_ABR = [x.replace("uni_","") for x in UNIS]
    UNIS_ABR = [x[0]+ ' Mov.' for x in UNIS_ABR]

    # MOVESET DICTS #################################################

    REPERTOIRE_ABR_DICTS = {
        CO_MOVESET: dict(zip(COS, COS_ABR)),
        UNI_MOVESET: dict(zip(UNIS, UNIS_ABR)),
    }

    # SIM_SETS ######################################################
    SIM_SET_DATA ={
        'centerout_rad': '_rad',
        'centerout_onehot': '_onehot',
        'uni_onehot': "_onehot",
        'uni_rad': "_rad",
        'uni_synth_rad': "_synth_rad",
    }
    
    SIM_SET_TASKS ={
        'centerout_rad': '_rad',
        'centerout_onehot': '_onehot',
        'uni_synth_rad': UNIS,
        'uni_rad': UNIS,
        'uni_onehot': UNIS,
    }

    # SIMULATION PARAMETERS ######################################################
    PERT_BATCH_SIZE = 64
    PERT_TRAINING_TRIALS = 100
    PERT_LR = 0.005
    PERT_LOG_INTERVAL = 10
    PRINT_EPOCH = 5
    
    # SIMULATION TYPES ######################################################
    PROJECT = "motor_learning"
    INIT_TRAINING = "init_training"
    PERTURBATION = "perturbation"

    ## PERT TYPES
    PERT_ROTATION = "rotation"
    # PERT_LENGTH = "length"
    PERT_REASSOCIATION = "reassoc"
    
    PERTURBATIONS = [PERT_ROTATION]
    PERT_PARAMS_DICT = {
        PERT_ROTATION: [10.0, 20.0,30.0,60.0],
    }
