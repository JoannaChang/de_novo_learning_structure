from typing import List, Optional, Tuple, Union


class Field:
    """Object to specify requirements for a field provided in configuration file."""
    def __init__(self, name: str, types: List, default = None, requirements: Optional[List] = None, key: Optional[str] = None):
        """
        Class constructor.

        Args:
            name: leaf-level name given to parameter/property in configuration file.
            key: name (ideally defined in a constants file) under which parameter 
            is stored in configuration object and subsequently retrieved with.
            If this is not provided, name will be used by default.
            types: list of valid types for property.
            default: default value if field does not exist.
            requirements: list of lambda functions to test validity of property.
        """
        self._name = name
        self._key = key or self._name
        self._types = tuple(types)
        self._default = default
        self._requirements = requirements

    @property
    def name(self) -> str:
        return self._name

    @property
    def key(self) -> str:
        return self._key

    @property
    def types(self) -> Tuple:
        return self._types
    
    @property
    def default(self):
        return self._default

    @property
    def requirements(self) -> Union[List, None]:
        return self._requirements