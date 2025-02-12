#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
During tracing, the function or class in this file might be wrapped as another function or class.
If the original function is needed to use (usually in tracer), should call the function in this file.
"""

# all functions in operator will be wrapped during tracing
from operator import *

# the wrapped functon/class in builtins
import builtins

isinstance = builtins.isinstance
issubclass = builtins.issubclass
len = builtins.len
getattr = builtins.getattr
id = builtins.id

bool = builtins.bool
int = builtins.int
float = builtins.float
frozenset = builtins.frozenset
tuple = builtins.tuple
list = builtins.list
set = builtins.set
dict = builtins.dict

enumerate = builtins.enumerate
map = builtins.map
range = builtins.range
reversed = builtins.reversed
type = builtins.type
slice = builtins.slice

all = builtins.all
min = builtins.min
max = builtins.max

# the wrapped functon/class method/class in torch
import torch

torch_module_getattr = torch.nn.Module.__getattr__
torch_module_getattribute = torch.nn.Module.__getattribute__
torch_module_call = torch.nn.Module.__call__
torch_agfunc_apply = torch.autograd.function.Function.apply
torch_assert = torch._assert
torch_Size = torch.Size
torch_finfo = torch.finfo
torch_autocast = torch.autocast

import importlib
import_module = importlib.import_module
