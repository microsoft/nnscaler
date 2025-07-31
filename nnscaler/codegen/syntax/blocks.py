#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional

class Block:

    def __init__(self, title):
        if not isinstance(title, str):
            raise TypeError(f"Expected string, but got {type(title)}")
        self.code = [title]

    def __enter__(self):
        return self

    def insert_body(self, code):
        if isinstance(code, list):
            self.code += code
        elif isinstance(code, str):
            self.code.append(code)
        else:
            raise TypeError(
                f"Get type {type(code)} but expected list[str] or list"
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        # add indent for function block
        for idx in range(1, len(self.code)):
            # use 4 space as indent
            self.code[idx] = '    ' + self.code[idx]
        if not exc_tb is None:
            print('Error detected in function block')


class FunctionBlock(Block):
    """
    Create a function block with function definition

    If class has derived class, then require the derived classes
    has no argument for __init__
    """

    def __init__(self, func_name: str, args: List[str], derived=True):
        if not isinstance(func_name, str):
            raise TypeError("Expected func_name to be str")
        if not isinstance(args, list):
            raise TypeError("Expcted args to be list[str]")
        self.func_name = func_name
        self.param_name = args
        args = ', '.join(args)
        title = f'def {self.func_name}({args}):'
        super().__init__(title)
        self.derived = derived

    def __enter__(self):
        # assume no argument for initialize super class
        if self.derived and self.func_name == '__init__':
            self.insert_body('super().__init__()')
        return self


class ClassBlock(Block):
    """
    Class definition.
    """

    def __init__(self, class_name, derived=None):
        if not isinstance(class_name, str):
            raise TypeError("Expected class_name to be str")
        if not isinstance(derived, list) and derived is not None:
            raise TypeError("Expcted derived to be None or list[str]")
        self.class_name = class_name
        if derived:
            derived = ', '.join(derived)
            derived = f'({derived})'
        title = f'class {self.class_name}{derived}:'
        super().__init__(title)


class ForBlock(Block):
    """
    Create a for-loop block with function definition
    """
    def __init__(self, var: Optional[str], iters: str):
        var = '_' if var is None else var
        super().__init__(f'for {var} in {iters}:')
