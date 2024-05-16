import numpy as np
import pyaldata as pyal
from matplotlib import cm

class Constants:
    # DIRECTORIES ##################################################
    PROJ_DIR="/home/jcc319/structure/" #change based on your directory structure
    RESULTS_FOLDER = "results/"
    DATA_FOLDER = "data/"
    EXP_DATA_FOLDER = "exp_data/"
    FIG_FOLDER = PROJ_DIR + 'figures/'
    PROCESSED_DATA_FOLDER = FIG_FOLDER + 'processed/'
    CONFIGS_DIR = "simulation/configs/"

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

    ## AMPA_MOVESET for tasks with 2 sinewaves with variable amplitude in first sinewave, fixed freq
    AMPA1_MOVESET = 'ampA1'
    AMPSA1 = [
        'ampA1_1.0',
        'ampA1_2movs_1_7',
        'ampA1_3movs_1_7',
        'ampA1_4movs_1_7',
        ]
    
    AMPSA1_ABR = [f'{x+1} Mov.' for x in range(4)]

    AMPB1_MOVESET = 'ampB1'
    AMPSB1 = [
        'ampB1_1.0',
        'ampB1_2movs_1_7',
        'ampB1_3movs_1_7',
        'ampB1_4movs_1_7',
        ]
    
    AMPSB_ABR = [f'{x+1} Mov.' for x in range(4)]
     

    # MOVESET DICTS #################################################

    REPERTOIRE_ABR_DICTS = {
        CO_MOVESET: dict(zip(COS, COS_ABR)),
        UNI_MOVESET: dict(zip(UNIS, UNIS_ABR)),
        AMPA1_MOVESET: dict(zip(AMPSA1, AMPSA1_ABR)),
        AMPB1_MOVESET: dict(zip(AMPSB1, AMPSB_ABR)),
    }

    # SIM_SETS ######################################################
    SIM_SET_DATA ={
        'centerout_rad': '_rad',
        'centerout_onehot': '_onehot',
        'uni_onehot': "_onehot",
        'uni_rad': "_rad",
        'uni_synth_rad': "_synth_rad",
        'uni_synth_fixed_rad': "_synth_fixed_rad",
        'uni_noprep_rad': "_noprep_rad",
        'uni_noprep_onehot': "_noprep_onehot",
        'ampA1_cont_onehot': "_cont_onehot",
        'ampB1_cont_onehot': "_cont_onehot",

    }
    
    SIM_SET_TASKS ={
        'centerout_rad': '_rad',
        'centerout_onehot': '_onehot',
        'uni_synth_rad': UNIS,
        'uni_synth_fixed_rad': UNIS,
        'uni_noprep_rad': UNIS,
        'uni_noprep_onehot': UNIS,
        'uni_rad': UNIS,
        'uni_onehot': UNIS,
        'ampA1_cont_onehot': AMPSA1,
        'ampB1_cont_onehot': AMPSB1,
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
    PERT_AMPLITUDE = "pertamp"
    PERT_AMPLITUDE_A = "pertampA"
    PERT_AMPLITUDE_B = "pertampB"
    PERT_FREQUENCY = "pertfreq"
    
    PERTURBATIONS = [PERT_ROTATION]
    PERT_PARAMS_DICT = {
        PERT_ROTATION: [10.0, 20.0,30.0,60.0],
    }
