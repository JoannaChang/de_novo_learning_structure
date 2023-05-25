import yaml

from config_manager import config_field
from config_manager import config_template
from config_manager import base_configuration

from tests import constants


class TestCase:

    _person_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.Constants.NUM_ARMS, types=[int], key=constants.Constants.NUM_ARMS, requirements=[lambda x: x > 0]),
            config_field.Field(name=constants.Constants.NUM_LEGS, types=[int], key=constants.Constants.NUM_LEGS, requirements=[lambda x: x > 0])
        ],
        level=[constants.Constants.PERSON]
    )

    _cat_template = config_template.Template(
        dependent_variables = [constants.Constants.ANIMAL_TYPE],
        dependent_variables_required_values = [[constants.Constants.CAT]],
        fields=[
            config_field.Field(name=constants.Constants.WHISKERS, types=[bool], key=constants.Constants.WHISKERS, requirements=[lambda x: x is True])
        ],
        level=[constants.Constants.ANIMAL, constants.Constants.CAT]
    )

    _dog_template = config_template.Template(
        dependent_variables = [constants.Constants.ANIMAL_TYPE],
        dependent_variables_required_values = [[constants.Constants.DOG]],
        fields=[
            config_field.Field(name=constants.Constants.WHISKERS, types=[bool], key=constants.Constants.WHISKERS, requirements=[lambda x: x is False])
        ],
        level=[constants.Constants.ANIMAL, constants.Constants.DOG]
    )

    _animal_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.Constants.TYPE, types=[str], key=constants.Constants.ANIMAL_TYPE, 
            requirements=[lambda x: x in [constants.Constants.CAT, constants.Constants.DOG]])
        ],
        level=[constants.Constants.ANIMAL],
        nested_templates = [_cat_template, _dog_template]
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.Constants.NAME, types=[str], key=constants.Constants.NAME),
            config_field.Field(name=constants.Constants.SURNAME, types=[str], key=constants.Constants.SURNAME),
            config_field.Field(name=constants.Constants.TYPE, types=[str], key=constants.Constants.TYPE, 
            requirements=[lambda x: x in [constants.Constants.PERSON, constants.Constants.ANIMAL]])
        ],
        nested_templates=[_person_template, _animal_template]
    )
