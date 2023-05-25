#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for running simulations.
"""
import time, os
import argparse
import datetime
import constants
from simulation.runner import Runner

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    """ Get command line arguments. """
    parser = argparse.ArgumentParser(description='Simulation parameters')
    parser.add_argument(
        'type', 
        type=str, 
        help = "job type",
        choices=["init_training", "perturbation"]
        )
    parser.add_argument(
        'seed', 
        type=int, 
        help = "random seed"
        )
    parser.add_argument(
        'sim_set', 
        type=str, 
        help = "simulation set",
        )
    parser.add_argument(
        'repertoire', 
        type=str, 
        help = "repertoire"
        )
    parser.add_argument(
        '-file', 
        type=str, 
        help = "data file",
        )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help="path to configuration file for simulations",
        default="config.yaml",
        )
    parser.add_argument(
        '-p', '--perturbation', 
        help = "perturbation", 
        default = None
        )
    parser.add_argument(
        '-pp', '--pert_params', 
        help = "perturbation params", 
        type=float,
        nargs = '+',
        default = [0]
        )
    parser.add_argument(
        '-t', '--test', 
        help = "if simulation is a test", 
        nargs='?', const='c'
        )
    # parser.add_argument(
    #     '-w', '--wandb', 
    #     help = "tag to use wandb", 
    #     nargs='?', const='c'
    #     )
    # parser.add_argument(
    #     '-v', '--vary', 
    #     help = "vary initial learning", 
    #     type = str
    #     )
    parser.add_argument(
        '-gpu', '--gpu_id', 
        type =int,
        help = "gpu to use", 
        )
    parser.add_argument(
        '-e', '--log_epochs', 
        type =int,
        help = "epochs for logging model", 
        nargs = '+'
        )
    parser.add_argument(
        '-i', '--log_interval', 
        type =int,
        help = "interval for logging model", 
        )
    parser.add_argument(
        '-tt', '--training_trials', 
        type =int,
        help = "number of training trials", 
        )
    parser.add_argument(
        '-lr', '--learning_rate', 
        type =float,
        help = "learning rate", 
        )
    parser.add_argument(
        '-n', '--sim_number', 
        type =int,
        help = "variation number for set of simulation", 
        )
    parser.add_argument(
        '-noise', 
        type =float,
        help = "noise to add to dynamics", 
        )
    # parser.add_argument(
    #     '-nonlin', 
    #     type =str,
    #     help = "nonlinearity in recurrent networks", 
    #     )
    parser.add_argument(
        '-o', '--optimizer', 
        type =str,
        help = "optimizer for training", 
        )
    parser.add_argument(
        '-fi', '--freeze_input', 
        help = "whether to freeze input weights", 
        nargs='?', const='c'
        )
    parser.add_argument(
        '-fr', '--freeze_rec', 
        help = "whether to freeze recurrent weights", 
        nargs='?', const='c'
        )

    args = parser.parse_args()

    return args

def set_random_seed(rand_seed):
    """ Set random seeds in places that use random generators. """
    import numpy as np
    import torch

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

def set_outdir(config,args):
    """ Set the output directory where results should be saved."""
    results_folder = constants.Constants.RESULTS_FOLDER
    seed_str = str(config.seed)

    if config.istest:
        outdir = results_folder + 'test/' + seed_str + '/' + config.repertoire + '/'
    else:
        outdir = results_folder + config.sim_set + '/' + seed_str + '/' + config.repertoire + '/' 

    if config.type == "perturbation":
        sim_num = 'v'+str(args.sim_number) + '/' if args.sim_number is not None else ""
        outdir = outdir + config.perturbation + '/' + sim_num + ("_".join(str(x) for x in config.pert_params)) + '/' 

    os.makedirs(outdir, exist_ok=True)
    # print("outdir:", outdir)

    config.amend_property(property_name="outdir", new_property_value=outdir)

    return config

def set_datadir(config):
    """ Set the directory where the data is located. """
    data_folder = constants.Constants.DATA_FOLDER
    if config.datafile is not None:
        datadir = data_folder + str(config.seed) + '/' + config.datafile
    else:
        dataset = constants.Constants.SIM_SET_DATA[config.sim_set]
        if config.perturbation == constants.Constants.PERT_REASSOCIATION:
            datadir = data_folder + str(config.seed) + '/' + config.repertoire + dataset + '_' + constants.Constants.PERT_REASSOCIATION
        else: 
            datadir = data_folder + str(config.seed) + '/' + config.repertoire + dataset
    print('datadir:', datadir)
    config.amend_property(property_name="datadir", new_property_value=datadir)
    return config

def set_sim_metadata(config, args):
    """ Set metadata specific to simulation. """
    config = set_outdir(config,args)
    config = set_datadir(config)

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    config.amend_property('timestamp', exp_timestamp)

    return config

def set_perturbation_config(config, args, training_dir):
    """ Set parameters for perturbation simulations. """
    if args.test is None:
        config.amend_property(property_name="training_trials", new_property_value=constants.Constants.PERT_TRAINING_TRIALS)
        print('tt', constants.Constants.PERT_TRAINING_TRIALS)
    config.amend_property(property_name="batch_size", new_property_value=constants.Constants.PERT_BATCH_SIZE)
    config.amend_property(property_name="lr", new_property_value=constants.Constants.PERT_LR)
    config.amend_property(property_name="perturbation", new_property_value=args.perturbation)
    config.amend_property(property_name="pert_params", new_property_value=args.pert_params)
    config.amend_property(property_name="type", new_property_value=args.type)
    config.amend_property(property_name="training_dir", new_property_value=training_dir)
    if config.log_interval is None and config.log_epochs is None:
        config.amend_property(property_name="log_interval", new_property_value=constants.Constants.PERT_LOG_INTERVAL)

    return config

def get_config(args):
    """ Get and edit the configuration object which specifies parameters for the simulation."""
    from config_manager import base_configuration
    from simulation.config_template import ConfigTemplate

    if args.type == constants.Constants.PERTURBATION:
        results_folder = constants.Constants.RESULTS_FOLDER
        training_dir = results_folder + args.sim_set + '/' + str(args.seed) + '/' + args.repertoire + '/'
        config_path = training_dir + 'config.yaml' 
    elif args.type == constants.Constants.INIT_TRAINING:
        config_path = os.path.join(MAIN_FILE_PATH, args.config)

    configuration = base_configuration.BaseConfiguration(
        configuration= config_path, template = ConfigTemplate.base_config_template)
    
    configuration.amend_property(property_name="seed", new_property_value=args.seed)
    configuration.amend_property(property_name="sim_set", new_property_value=args.sim_set)
    configuration.amend_property(property_name="repertoire", new_property_value=args.repertoire)

    if args.log_epochs:
        configuration.amend_property(property_name="log_epochs", new_property_value=args.log_epochs)
    if args.log_interval:
        configuration.amend_property(property_name="log_interval", new_property_value=args.log_interval)
    if args.type == constants.Constants.PERTURBATION:
        configuration = set_perturbation_config(configuration, args, training_dir)
    if args.test:
        configuration.amend_property(property_name="istest", new_property_value=True)
        configuration.amend_property(property_name="batch_size", new_property_value=constants.Constants.TEST_BATCH_SIZE)
        configuration.amend_property(property_name="training_trials", new_property_value=constants.Constants.TEST_TRAINING_TRIALS)
    # if args.wandb:
    #     configuration.amend_property(property_name="wandb", new_property_value=True)
    if args.freeze_input:
        configuration.amend_property(property_name="freeze_input", new_property_value=True)
    if args.freeze_rec:
        configuration.amend_property(property_name="freeze_rec", new_property_value=True)
    if args.gpu_id:
        configuration.amend_property(property_name="gpu_id", new_property_value=args.gpu_id)
#         print("gpu", args.gpu_id)
    if args.file:
        configuration.amend_property(property_name="datafile", new_property_value=args.file)
    if args.training_trials:
        configuration.amend_property(property_name="training_trials", new_property_value=args.training_trials)
    if args.learning_rate:
        configuration.amend_property(property_name="lr", new_property_value=args.learning_rate)
    if args.noise:
        configuration.amend_property(property_name="noise", new_property_value=args.noise)
    # if args.nonlin:
    #     configuration.amend_property(property_name="nonlin", new_property_value=args.nonlin)
    if args.optimizer:
        configuration.amend_property(property_name="optimizer", new_property_value=args.optimizer)
        if args.optimizer == 'FORCE':
            configuration.amend_property(property_name="batch_size", new_property_value=1)

    #FORCE training with 1 trial at a time
    if configuration.optimizer == 'FORCE':
        configuration.amend_property(property_name="batch_size", new_property_value=1)

    return configuration

def run(config):
    """ Set up and train simulation model. """
    runner = Runner(config, os.getcwd())
    runner.run_train()

if __name__ == "__main__":
    import time

    starttime = time.time()

    args = get_args()
    set_random_seed(args.seed)
    config = get_config(args)
    config = set_sim_metadata(config, args)
    run(config)

    endtime = time.time()
    print('Total time: %.2f'%(endtime-starttime))
