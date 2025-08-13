#!/usr/bin/env python3

"""
Setup script for nnscaler with C++ extensions
"""

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup

# Define C++ extensions
def get_ext_modules():
    """Get extension modules with appropriate compiler flags"""
    
    # Base compile args
    compile_args = ['-O3', '-fPIC']
    
    # Try to use older ABI for better compatibility (following PyTorch's approach)
    compile_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
    
    # Link arguments
    link_args = ['-lpthread']
    
    # conda environment handling, since:
    # - nnscaler may be installed in conda, for example, user's development environment and our ci.
    # - libstdc++ in conda may be different from system libstdc++.
    # - we prefer the conda version for compatibility.
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        # Fallback to ANACONDA_PYTHON_VERSION like PyTorch does
        anaconda_python_version = os.environ.get('ANACONDA_PYTHON_VERSION')
        if anaconda_python_version:
            conda_prefix = f"/opt/conda/envs/py_{anaconda_python_version}"
    
    if conda_prefix:
        # Add conda library path with RPATH for runtime discovery
        conda_lib_path = os.path.join(conda_prefix, 'lib')
        if os.path.exists(conda_lib_path):
            link_args.extend([f'-L{conda_lib_path}', f'-Wl,-rpath,{conda_lib_path}'])
    
    ext_modules = [
        Pybind11Extension(
            "nnscaler.autodist.dp_solver",
            [
                "nnscaler/autodist/dp_solver.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
                "nnscaler/autodist",
            ],
            language='c++',
            cxx_std=11,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ]
    
    return ext_modules

# Custom build_ext class to provide feedback
class CustomBuildExt(build_ext):
    """Custom build extension to handle C++ compilation"""
    
    def build_extensions(self):
        print("Building C++ extensions...")
        for ext in self.extensions:
            print(f"  - {ext.name}")
        
        # Print environment info
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            print(f"  Using conda environment: {conda_prefix}")
        
        super().build_extensions()
        print("C++ extensions built successfully!")

setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
