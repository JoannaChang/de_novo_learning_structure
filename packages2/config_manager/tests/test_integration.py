import yaml
import os

import unittest

from config_manager import base_configuration
from config_manager import config_field

from tests.test_templates import test_template_1

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class TestIntegration(unittest.TestCase):
    """Test class for integration of functionality in config_manager package."""
    def _test_integration_example(self, configuration, template):
        bc = base_configuration.BaseConfiguration(configuration=configuration, template=template)

    def test_integration_examples(self):
        test_templates = [test_template_1.TestCase.base_config_template]
        for i in range(1):
            yaml_file_path = os.path.join(FILE_PATH, f"test_configs/test_config_{i + 1}.yaml")
            with open(yaml_file_path, 'r') as yaml_file:
                configuration = yaml.load(yaml_file, yaml.SafeLoader)
            self._test_integration_example(configuration, test_templates[i])
            

def get_suite():
    model_tests = [
        'test_integration_examples'
        ]
    return unittest.TestSuite(map(TestIntegration, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=1)
runner.run(get_suite())
