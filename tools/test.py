#%%
from config_manager import base_configuration
from simulation.config_template import ConfigTemplate
import constants
from simulation.runner import Runner
import os
import contextlib
import numpy as np

def set_random_seed(rand_seed:int):
    import numpy as np
    import torch

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

def setup_runner(config_dir:str, datafile:str = None, noise:float = None, training:bool=False):
    """
    Set up a runner for a model.

    Parameters
    ----------
    config_dir: directory where model is stored
    datafile: data to train/test the model on
    noise: noise to include in network; if None, use noise from config file
    training: whether to train the model or not
    """

    config_path = config_dir + "config.yaml"
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        config = base_configuration.BaseConfiguration(
            configuration= config_path, template = ConfigTemplate.base_config_template)
        
    if datafile is not None:
        data_folder = constants.Constants.DATA_FOLDER
        datadir = data_folder + str(config.seed) + '/' + datafile
        config.amend_property(property_name="datadir", new_property_value=datadir)
    if noise is not None:
        config.amend_property(property_name="noise", new_property_value=noise)
    
    set_random_seed(config.seed)

    runner = Runner(config, constants.Constants.PROJ_DIR, training = training)

    return runner
    
def test_model(config_dir:str, epoch:int = None, datafile:str = None, r1_input:np.array = None, noise:float = None):
    """
    Test a trained model.

    Parameters
    ----------
    config_dir: directory where model is stored
    epoch: epoch model was at during training
    datafile: data to test the model on
    r1_input: pre-defined activity for the first RNN (trials x tsteps x neurons)     
    noise: noise to include in network; if None, use noise from config file

    """
    runner = setup_runner(config_dir, datafile, noise)
    datadir, output_pre, output, activity1 = runner.run_test(epoch, r1_input)
    return datadir, output_pre, output, activity1

# def get_hidden_state(config_dir:str, epoch:int = None, datafile:str = None, noise:float = None):
#     """
#     Get hidden state from testing a trained model.

#     Parameters
#     ----------
#     config_dir: directory where model is stored
#     epoch: epoch model was at during training
#     datafile: data to test the model on
#     noise: noise to include in network; if None, use noise from config file
#     """
#     runner = setup_runner(config_dir, datafile, noise)
#     hidden1 = runner.get_hidden_from_test(epoch)
#     return hidden1


# %%
