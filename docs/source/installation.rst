############
Installation
############

nnScaler can either be installed from the wheel package or from the source code.

******************
Install from Wheel
******************

The wheel package is hosted on `GitHub release <https://github.com/microsoft/nnscaler/releases>`_.

.. code-block:: bash

    pip install https://github.com/microsoft/nnscaler/releases/download/0.5/nnscaler-0.5-py3-none-any.whl

************************
Install from Source Code
************************

Editable Install
================

nnScaler uses ``pybind11`` and ``cppimport`` to dynamically build C++ modules.
The C++ modules must be manually compiled for an editable install.

.. code-block:: bash

    git clone --recursive https://github.com/microsoft/nnscaler
    cd nnscaler
    pip install -e .
    python -c "import cppimport.import_hook ; import nnscaler.autodist.dp_solver"

Build a Wheel
=============

Alternatively you can build the wheel package by yourself.

.. code-block:: bash

    cd nnscaler
    pip install build
    python -m build
    pip install dist/nnscaler-*.whl
