#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Pseudo module of resource files.
"""

__all__ = 'files'

import importlib.resources

def files():
    """
    Alias of ``importlib.resources.files('nnscaler.resources')``.

    Returns:
        A ``Path``-like object.

    Example:
        ::
            import nnscaler.resources
            (nnscaler.resources.files() / 'path/to/my_file.txt').read_text()
    """
    return importlib.resources.files(__name__)
