#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Pseudo module of resource files.
"""

from __future__ import annotations

__all__ = 'files'

# TODO: when drop python 3.8 support, change it to `importlib.resources`
import importlib_resources
from importlib_resources.abc import Traversable

def files() -> Traversable:
    """
    Alias of ``importlib.resources.files('nnscaler.resources')``.

    Returns:
        A ``Path``-like object.

    Example:
        ::
            import nnscaler.resources
            (nnscaler.resources.files() / 'path/to/my_file.txt').read_text()
    """
    return importlib_resources.files(__name__)
