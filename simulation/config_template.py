
from config_manager import config_field
from config_manager import config_template
import constants 

class ConfigTemplate:

    _simulation_template = config_template.Template(
        fields=[
            config_field.Field(
                name='type',
                types=[str],
                requirements=[
                    lambda x: x in [constants.Constants.INIT_TRAINING, constants.Constants.PERTURBATION]
                ],
            ),
            config_field.Field(
                name='sim_set',
                types=[str],
                requirements=[
                    lambda x: x in [
                        'centerout_rad',
                        'centerout_onehot',
                        'uni_synth_rad',
                        'uni_rad',
                        'uni_onehot',
                    ]
                ],
            ),
            config_field.Field(
                name='repertoire',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='istest',
                types=[bool],
            ),
            config_field.Field(
                name='vary',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='training_dir',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='network',
                types=[str],
                requirements=[
                    lambda x: x in ['RNN_single']
                ],
            ),
            config_field.Field(
                name='nonlin',
                types=[str],
                requirements=[
                    lambda x: x in ['relu', 'tanh']
                ],
            ),
            config_field.Field(
                name='optimizer',
                types=[str],
                requirements=[
                    lambda x: x in ['Adam', 'SGD', 'FORCE']
                ],
            ),
            config_field.Field(
                name='freeze_input',
                types=[bool],
            ),
            config_field.Field(
                name='freeze_rec',
                types=[bool],
            ),
        ],
        level=['simulation'],
    )

    _perturbation_template = config_template.Template(
        fields=[
            config_field.Field(
                name='perturbation',
                types=[str, type(None)],
                requirements=[
                    lambda x: x in [
                        None, 
                        constants.Constants.PERT_ROTATION,
                        # constants.Constants.PERT_LENGTH,
                        constants.Constants.PERT_REASSOCIATION,
                        ]
                ],
            ),
            config_field.Field(
                name='pert_params',
                types=[list],
            ),
        ],
        level=['perturbation'],
    )

    _data_template = config_template.Template(
        fields=[
            config_field.Field(
                name='datadir',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='datafile',
                types=[str, type(None)],
            ),
        ],
        level=['data'],
    )

    _neurons_template = config_template.Template(
        fields=[
            config_field.Field(
                name='n1',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='tau',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='g1',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='gin',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='gout',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='noise',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
        ],
        level=['neurons'],
    )

    _regularization_template = config_template.Template(
        fields=[
            config_field.Field(
                name='alpha1',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='gamma1',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='beta1',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='clipgrad',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
        ],
        level=['regularization'],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name='batch_size',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='training_trials',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='lr',
                types=[float],
                requirements=[
                    lambda x: x > 0 
                ],
            ),
        ],
        level=['training'],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name='log_model',
                types=[bool],
            ),
            config_field.Field(
                name='log_interval',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='log_epochs',
                types=[list, type(None)],
            ),
        ],
        level=['logging'],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(
                name='outdir',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='gpu_id',
                types=[int],
            ),
            config_field.Field(
                name='seed',
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name='timestamp',
                types=[str, type(None)],
            ),
        ],
        nested_templates=[
            _simulation_template,
            _perturbation_template,
            _data_template,
            _neurons_template,
            _regularization_template,
            _training_template,
            _logging_template,
        ],
    )