## Configuration Manager

This module is designed for managing a configuration for computational simulations and experimentation (e.g. machine learning experiments). Along with a configuration file (yaml format), user must specify _template_ objects containing _fields_ (and/or further nested templates) that specify the structure expected of the configuration file. Each _field_ object specifies the types required for the configuration entry as well as a set of lambda functions that can check for other conditions (e.g. value ranges or set membership). When the yaml file is read into the main configuration object, type and other requirement checks are made; subsequently the configuration parameters are stored as attributes in the instance of the configuration object, the names of which are specified in the _key_ variable of the _field_ objects making up the template.

Requirements are minimal: PyYAML.

Setup: clone repository and run ```pip install -e .``` from root. 