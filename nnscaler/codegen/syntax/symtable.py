#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

class SymbolTable:
    """
    Symbolic table for saving declared variables.

    Assume the program will first declare all possible used
    variables before entering any of its sub (children) scopes.

    Attributes:
        name (str): name of this scope
        _varlist (dict{str: DType}): declared variable dict
            var_name -> type_of_var
    """

    def __init__(self):
        self._varlist = list()

    def create(self, var_name: str):
        """
        Create a variable.

        Args:
            var_name (str): variable name

        Returns:
            True if declared, False if the var already exists.
        """
        assert isinstance(var_name, str)
        if var_name in self._varlist:
            return False
        else:
            self._varlist.append(var_name)
            return True

    def exist(self, var_name: str):
        """
        Check whether a variable exists
        """
        assert isinstance(var_name, str)
        if var_name in self._varlist:
            return True
        else:
            return False
