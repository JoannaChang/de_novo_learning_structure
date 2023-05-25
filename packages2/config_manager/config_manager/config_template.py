from abc import ABC, abstractmethod

from typing import List, Union, Optional

from config_manager import config_field


class Template:
    """Object to specify required structure of configuration file."""
    def __init__(
        self, 
        fields: List[config_field.Field], 
        nested_templates: Optional[List] = None, 
        level: Optional[List[str]] = None, 
        dependent_variables: Optional[List[str]] = None, 
        dependent_variables_required_values: Optional[List[List]] = None
    ):
        """
        Class constructor.

        Args:
            fields: list of field objects expected to be present at this level of the configuration.
            nested_templates: list of fields at this level of configuration that are themselves groups of fields, 
                and require subsequent template.
            level: description of nesting in configuration.
            dependent_variables: (optional) list of configuration keys on which necessity of validating this template is dependent.
            dependent_variables_required_values: (required if dependent_variables is provided) 

        Raises:
            AssertionError: if dependent_variables are provided without dependent_variables_required_values
            AssertionError: if length of dependent_variables and dependent_variables_reqired_values do not match.
        """
        self._fields = fields
        self._nested_templates = nested_templates
        self._level = level

        if dependent_variables is not None:
            missing_error = f"Required values for dependent variables for template at level {level} missing."
            len_mismatch_error = (
                    f"Mismatch: {len(dependent_variables)} dependent_variables provided."
                    f"{len(dependent_variables_required_values)} sets of required_values provided.")
            assert dependent_variables_required_values is not None, missing_error
            assert len(dependent_variables) == len(dependent_variables_required_values), len_mismatch_error

        self._dependent_variables = dependent_variables
        self._dependent_variables_required_values = dependent_variables_required_values

    @property
    def fields(self) -> List[config_field.Field]:
        return self._fields

    @property
    def nested_templates(self) -> List:
        return self._nested_templates or []

    @property
    def level(self) -> Union[List[str], None]:
        return self._level

    @property
    def dependent_variables(self) -> Union[List[str], None]:
        return self._dependent_variables

    @property
    def dependent_variables_required_values(self) -> Union[List[List], None]:
        return self._dependent_variables_required_values

    @property
    def template_name(self) -> Union[str, None]:
        if self._level:
            return self._level[-1]
        else:
            return None
