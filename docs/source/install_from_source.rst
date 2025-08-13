###################
Install from Source
###################

**************
Clone the Repo
**************

The nnScaler repository is hosted on GitHub.

::

    git clone https://github.com/microsoft/nnscaler

****************
Editable Install
****************

nnScaler uses ``pybind11`` and ``cppimport`` to speedup partitioning.
The c++ modules must be manually compiled for an editable install.

::

    cd nnscaler
    pip install -e .
    python -c "import cppimport.import_hook ; import nnscaler.autodist.dp_solver"

*************
Build a Wheel
*************

::

    cd nnscaler
    pip install build
    python -m build
    pip install dist/nnscaler-*.whl
