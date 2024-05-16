# %%
import os
from constants import Constants
from config_manager import base_configuration
from simulation.config_template import ConfigTemplate
import numpy as np

# %%
# MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def create_config(config_num, property_name = None, property_value = None, changes_dict = None, base_config=None):
    assert(((property_name is not None) & (property_value is not None)) or (changes_dict is not None))
    if base_config is None:
        base_config = 'config.yaml'
    config_path = os.path.join(Constants.PROJ_DIR, 'simulation/configs/' + base_config)

    configuration = base_configuration.BaseConfiguration(
        configuration= config_path, template = ConfigTemplate.base_config_template)
    
    configuration.amend_property(property_name='config_number', new_property_value=config_num)

    if changes_dict is None:
        configuration.amend_property(property_name=property_name, new_property_value=property_value)
    else:
        for property_name in changes_dict.keys():
            property_value = changes_dict[property_name]
            configuration.amend_property(property_name=property_name, new_property_value=property_value)
    
    save_dir = Constants.PROJ_DIR + Constants.CONFIGS_DIR
    file_name = 'config_' + str(config_num) + '.yaml'
    configuration.save_configuration(folder_path=save_dir, file_name = file_name)


# %%
n_simulation = 0

#nonlinearity
create_config(1,'nonlin', 'relu')

#n1
for i, val in enumerate([100,500]):
    create_config(2 + i,'n1', val)

#g1
for i, val in enumerate([0.8, 1.6]):
    create_config(4 + i,'g1', float(val))

#gin
for i, val in enumerate([0.01, 1.0]):
    create_config(6 + i,'gin', float(val))

#gout
for i, val in enumerate([0.01, 1.0]):
    create_config(8 + i,'gout', float(val))

#noise
for i, val in enumerate([0.0, 0.4]):
    create_config(10 + i,'noise', float(val))

#alpha1 & gamma1
for i, val in enumerate([0.0001, 0.01]):
    config_dict = {'gamma1': float(val), 'alpha1': float(val)}
    create_config(12 + i, changes_dict = config_dict)

# #gamma1 
# for i, val in enumerate([0.001, 0.01]):
#     create_config(14 + i,'gamma1', float(val))

#beta1
for i, val in enumerate([0.2, 0.8]):
    create_config(16 + i,'beta1', float(val))

#training_trials
for i, val in enumerate([500, 1000]):
    create_config(18 + i,'training_trials', val)

#batch_size
for i, val in enumerate([32, 128]):
    create_config(20 + i,'batch_size', val)

#lr
for i, val in enumerate([0.00001, 0.001]):
    create_config(22 + i,'lr', float(val))

# %%
