import unittest

from config_manager import base_configuration
from config_manager import config_field


class TestBaseConfiguration(unittest.TestCase):
    """Test class for base_configuration.BaseConfiguration unit tests."""
    def test_validate_field(self):
        valid_test_fields = [
            ({"hello": "world"}, config_field.Field(name="hello", key="hello", types=[str], requirements=[lambda x: "or" in x])),
            ({"hello": "world"}, config_field.Field(name="hello", key="hello", types=[str, int], requirements=[lambda x: x == "world"])),
            ({"hello": "world"}, config_field.Field(name="hello", key="hello", types=[int, str], requirements=[lambda x: x.islower()]))
        ]

        invalid_test_fields = [
            ({"hello": 5}, config_field.Field(name="hello", key="hello", types=[str], requirements=[lambda x: "or" in x])),
            ({"hello": "world"}, config_field.Field(name="hello", key="hello", types=[str, int], requirements=[lambda x: x == "worl"])),
            ({"hello": 8}, config_field.Field(name="hello", key="hello", types=[int, str], requirements=[lambda x: x < 5]))
        ]

        for data, field in valid_test_fields:
            base_configuration.BaseConfiguration.validate_field(field=field, data=data, level='test')

        for data, field in invalid_test_fields:
            with self.assertRaises(AssertionError):
                base_configuration.BaseConfiguration.validate_field(field=field, data=data, level='test')


def get_suite():
    model_tests = [
        'test_validate_field'
        ]
    return unittest.TestSuite(map(TestBaseConfiguration, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=1)
runner.run(get_suite())
