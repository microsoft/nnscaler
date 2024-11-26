#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import operator
from typing import Any, Callable, Set

import torch
from torch.nn import Sequential, ModuleList, ModuleDict
from torch.fx.node import _side_effectful_functions, Node

from . import wrap_utils


side_effectful_inplace_ops = {
    operator.iadd, operator.isub, operator.imul, operator.itruediv, operator.ifloordiv,
    operator.iand, operator.ior, operator.ixor, operator.ilshift, operator.irshift,
    operator.imod, operator.ipow,
    # operator.imatmul is not implemented in torch
    # so let's ignore it now
    operator.setitem,
}


class ExtraSEFPatcher:
    def __init__(self, extra_side_effectful_functions: Set[Callable]):
        self.extra_side_effectful_functions = extra_side_effectful_functions
        self.incontext_funcs = set()

    def __enter__(self):
        self.incontext_funcs = self.extra_side_effectful_functions - _side_effectful_functions
        _side_effectful_functions.update(self.incontext_funcs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _side_effectful_functions.difference_update(self.incontext_funcs)


def is_useless_iter(node: Node):
    if node.op == 'call_function' and node.target is iter:
        node_is_impure = False
        for iter_user in node.users:
            if not is_useless_next(iter_user):
                node_is_impure = True
                break
        if not node_is_impure:
            for iter_user in list(node.users.keys()):
                setattr(iter_user, '_is_impure', False)
                iter_user.graph.erase_node(iter_user)
            if len(node.users) > 0:
                raise RuntimeError('The user node of iter is not empty, something goning wrong.')
            setattr(node, '_is_impure', False)
            return True
    else:
        return False


def is_useless_next(node: Node):
    if node.op == "call_function" and node.target is next:
        if len(node.users) == 0:
            return True
    else:
        return False


class TorchFXPatcher:
    """
    this patcher is a context mananger, when enter the context, several torch.fx functions will be patched,
    and revert these functions when exit.
    
    The following function will be patched:
    
    torch.fx.graph.magic_methods:
        additional add not_/is_/is_not/contains, because these functions are transformed by nnscaler operator patcher.

    torch.fx.graph_module._copy_attr:
        additional track persistent attribute for buffer.

    torch.fx.graph_module._format_import_statement:
        additional support autograd functions code generation.

    torch.fx.node._find_module_of_method:
        additional support autograd functions and _VariableFunctionsClass functions for find the correct module.

    torch.fx.node.is_impure:
        additional add inplace functions as impure nodes and useless iter nodes as non-impure node.
    """
    from torch.fx import graph as fx_graph
    from torch.fx import graph_module as fx_graph_module
    from torch.fx import node as fx_node

    magic_methods_ori = fx_graph.magic_methods
    copy_attr_ori = fx_graph_module._copy_attr
    find_module_of_method_ori = fx_node._find_module_of_method
    is_impure_ori = fx_node.Node.is_impure
    format_import_statement_ori = fx_graph_module._format_import_statement    

    magic_methods_new = {
        **fx_graph.magic_methods,
        # NOTE by nnscaler: add these method because we use operator patcher to transform the origin code to `_operator.xxx`,
        # torch.fx.graph.magic_methods is used to emit node to generate code, so here add these mapping to make the gencode more readable. 
        # for example:
        #     in original code:         if mask is not None:
        #     will be transformed to:   if _operator.is_not(mask, None):
        'not_': 'not {}',
        'is_': '{} is {}',
        'is_not': '{} is not {}',
        'contains': '{1} in {0}',
    }

    @staticmethod
    def copy_attr_new(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
        """
        copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
        This installs empty Modules where none exist yet if they are subpaths of target
        """
        *prefix, field = target.split('.')
        for item in prefix:
            f = getattr(from_module, item)
            t = getattr(to_module, item, None)
            if f is t:
                # we have already installed one of its parents
                # (e.g. target = root.linear.weight, but we have already installed root.linear)
                # once we have installed a parent, we no longer need to copy the children
                # since all needed attributes have been copied
                return

            if t is None:
                # NOTE by nnscaler: in the original copy_attr, only create torch.nn.Module for all cases,
                # here we add more kinds of official subclasses of torch.nn.Module
                if isinstance(f, Sequential):
                    t = Sequential()
                elif isinstance(f, ModuleList):
                    t = ModuleList()
                elif isinstance(f, ModuleDict):
                    t = ModuleDict()
                else:
                    t = torch.nn.Module()
                # NOTE by nnscaler: for readable reason, we want the to_module has the same repr with the from_module,
                # so here we bind the from_module._get_name to to_module._get_name
                if hasattr(f, '_get_name'):
                    t._get_name = f._get_name
                to_module.add_module(item, t)
            from_module, to_module = f, t

        orig = getattr(from_module, field)
        
        # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
        # So, we register it as a named buffer in the target module.
        if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
            # NOTE by nnscaler: persistent state is not considered by the original copy_attr, so add it here
            persistent = field in from_module._buffers and field not in from_module._non_persistent_buffers_set
            to_module.register_buffer(field, orig, persistent=persistent)
        else:
            setattr(to_module, field, orig)

    @staticmethod
    def find_module_of_method_new(orig_method: Callable[..., Any]) -> str:
        # NOTE by nnscaler: if the method is torch.autograd.Function.apply, we should return its name with bound module
        # for example, cus_module.CusAutogradFunction is a class inherit the torch.autograd.Function, then:
        #     cus_module.CusAutogradFunction.apply.__name__ is "apply"
        #     cus_module.CusAutogradFunction.apply.__module__ is "torch.autograd.function"
        #     cus_module.CusAutogradFunction.apply.__self__.__name__ is "CusAutogradFunction"
        #     cus_module.CusAutogradFunction.apply.__self__.__module__ is "cus_module"
        # so the correct module path of the autograd apply method is f'{orig_method.__self__.__module__}.{orig_method.__self__.__name__}'
        if wrap_utils.is_autograd_apply(orig_method):
            return f'{orig_method.__self__.__module__}.{orig_method.__self__.__name__}'

        name = orig_method.__name__
        module = orig_method.__module__

        if module is not None:
            return module
        # NOTE by nnscaler: add a special support for torch._C._VariableFunctions
        if hasattr(orig_method, '__qualname__') \
            and isinstance(orig_method.__qualname__, str) and orig_method.__qualname__.startswith('_VariableFunctionsClass.'):
            return 'torch._C._VariableFunctions'
        for guess in [torch, getattr(torch.nn, 'functional')]:
            if getattr(guess, name, None) is orig_method:
                return guess.__name__
        raise RuntimeError(f'cannot find module for {orig_method}')

    @staticmethod
    def format_import_statement_new(name: str, obj: Any, importer) -> str:
        # NOTE by nnscaler: to support code generation of autograd function in nnscaler
        # for example:
        #   => input:  name=model_layer_CustomizedAutogradFunc_apply, obj=CustomizedAutogradFunc.apply
        #   => obj.__self__ is CustomizedAutogradFunc
        #   => return: from xxx import CustomizedAutogradFunc as model_layer_CustomizedAutogradFunc_apply
        #              model_layer_CustomizedAutogradFunc_apply = model_layer_CustomizedAutogradFunc_apply.apply
        if wrap_utils.is_autograd_apply(obj):
            return TorchFXPatcher.format_import_statement_ori(name, obj.__self__, importer) + f'\n{name} = {name}.apply'
        return TorchFXPatcher.format_import_statement_ori(name, obj, importer)

    @staticmethod
    def is_impure_new(node: fx_node.Node):
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Returns:

            bool: If the op is impure or not.
        """
        if is_useless_iter(node):
            return False

        if node.op in {"placeholder", "output"}:
            return True

        # Check if an impure function.
        if node.op == "call_function":
            return node.target in _side_effectful_functions

        # NOTE by nnscaler: we assume all method end with "_" is inplace operation,
        # and we take all inplace operations impure.
        if node.op == "call_method":
            return node.target.endswith("_")

        # Check if an impure module.
        if node.op == "call_module":
            assert (
                node.graph.owning_module is not None
            ), "self.graph.owning_module not set for purity check"
            target_mod = node.graph.owning_module.get_submodule(node.target)
            assert (
                target_mod is not None
            ), f"Did not find expected submodule target {node.target}"
            return getattr(target_mod, "_is_impure", False)

        return False

    def __enter__(self):
        TorchFXPatcher.fx_graph.magic_methods = self.magic_methods_new
        TorchFXPatcher.fx_graph_module._copy_attr = self.copy_attr_new
        TorchFXPatcher.fx_node._find_module_of_method = self.find_module_of_method_new
        TorchFXPatcher.fx_node.Node.is_impure = self.is_impure_new
        TorchFXPatcher.fx_graph_module._format_import_statement = self.format_import_statement_new
        TorchFXPatcher.available = True

    def __exit__(self, exc_type, exc_value, tb):
        TorchFXPatcher.fx_graph.magic_methods = TorchFXPatcher.magic_methods_ori
        TorchFXPatcher.fx_graph_module._copy_attr = TorchFXPatcher.copy_attr_ori
        TorchFXPatcher.fx_node._find_module_of_method = TorchFXPatcher.find_module_of_method_ori
        TorchFXPatcher.fx_node.Node.is_impure = TorchFXPatcher.is_impure_ori
        TorchFXPatcher.fx_graph_module._format_import_statement = TorchFXPatcher.format_import_statement_ori
        TorchFXPatcher.available = False
        return exc_type is None
