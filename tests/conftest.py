#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from pathlib import Path

from nnscaler.graph.parser import FxModuleParser

@pytest.fixture(autouse=True)
def clean_generated_files():
    print('hello')
    yield
    # try to clean generated files after each test run.
    basedir = Path('./').resolve()
    generated_files = [FxModuleParser.ATTR_CONTENT_FILE_0, FxModuleParser.ATTR_MAP_FILE]
    for f in generated_files:
        f = basedir / f
        if f.exists():
            f.unlink()
    for f in basedir.glob('gencode*.py'):
        f.unlink()
