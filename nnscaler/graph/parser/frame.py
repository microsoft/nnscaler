#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import OrderedDict
from typing import List, Any, Dict, Tuple, Optional
from nnscaler.ir.cten import IRTensor
import torch


class Frame:
    """
    Frame to save call stack and variable
    """
    def __init__(self):

        # var name -> value (IRTesnor, deterministic)
        self._vars: List[dict[str, Any]] = list()
        self._var_stack: List[str] = list()
        # IRTensor -> (module param name, concrete value)
        self._attr_map: Dict[IRTensor, Tuple[str, torch.Tensor]] = dict()

    def push_var(self, inherit_from_top=False):
        """
        Push a new variable frame as current variable frame.
        This should only be called when stepping in a module or method.

        Args:
            inherit_from_top (bool):
                whether to make all already defined variables in the top frame
                accessible to the evaluation procedure
                (e.g. references to such variables won't cause VarNotFound exception).
        """
        if inherit_from_top:
            assert len(self._vars) > 0
            self._vars.append(self._vars[-1].copy())
        else:
            self._vars.append(OrderedDict())

    def pop_var(self):
        """
        Pop the current variable frame.
        This should only be called when steping out a module or method.
        """
        if len(self._vars) == 0:
            raise RuntimeError("Try to pop stack with 0 depth")
        self._vars.pop()

    def add_var(self, var_name: str, val: Any, graph_arg: int = -1):
        """
        Add variable to the current frame

        Args:
            var_name (str): variable name (unique)
            val: variable content
            graph_arg (int):
                indicate whether it is an argument of the graph. -1 indicates not an argument.
                If >= 0, is a graph arg, will try to find val from variable stack,
                and link the name of the argument name from the callee function
                to the names of the argument passed-in.
        """

        if not isinstance(var_name, str):
            raise RuntimeError("Expected var_name is str")
        if var_name in self._vars[-1]:
            raise KeyError("Try to insert an already existed variable")
        # not a function parameter, no need for mapping
        if graph_arg == -1:
            self._vars[-1][var_name] = val
        # a function parameter, may need for mapping
        elif graph_arg >= 0:
            # root graph entry
            if self.depth() == 1:
                self._vars[-1][var_name] = val
            # fucnton call
            else:
                prev_frame = self._vars[-2]
                param_name = self._var_stack[-1-graph_arg]
                val = prev_frame[param_name]
                self._vars[-1][var_name] = val
        else:
            raise ValueError("graph_arg (int) must be >= 0")

    def del_val(self, var_name: str):
        """
        Delete a variable from the current frame.
        Do nothing if the variable doesn't exist.

        Args:
            var_name (str): variable name
        """
        self._vars[-1].pop(var_name, None)

    def set_var(self, var_name: str, val: Any):
        """
        Reset a variable with arbitrary value.
        If `var_name` doesn't exist, will create a new one

        @param var_name str: variable name
        @param val Any
        """
        self._vars[-1][var_name] = val

    def get_var(self, var_name: str) -> Any:
        """
        Get variable value according to var_name

        Special mapping between frames (function calls):

            input.x will be mapped to output.k at the about 1-hop frame

        Returns:
            val (Any)
        """
        # first check whether we have variable in this frame
        if var_name in self._vars[-1]:
            return self._vars[-1][var_name]
        # See rule 8 in graph/function/functions.py
        raise KeyError(
            f"Cannot find var name {var_name} in {self._vars}. "
            f"Please check whether the variable is from a function that is annotated as no-output."
        )

    def add_attr(self, tensor: IRTensor, concrete_value: torch.Tensor, name: str):
        """Add module attribute content

        Args:
            tensor (IRTensor): the tensor represents the value
            value (torch.Tensor or Any): concrete value
            name (str): attributed name of its original module
        """
        assert isinstance(concrete_value, torch.Tensor)
        self._attr_map[tensor] = (name, concrete_value)

    def get_attr_var(self, concrete_value: torch.Tensor) -> Optional[IRTensor]:
        """Get IRTensor from attribute concrete value

        If the concrete value is not found, return None
        """
        assert isinstance(concrete_value, torch.Tensor)
        for tensor, (_, value) in self._attr_map.items():
            if value is concrete_value:
                return tensor
        return None

    def save_attr_content(self, save_file_stem: str, params_per_file: int = 1024 * 1024 * 1024):
        """
        Save attribute content into file.

        Args:
            save_file_stem (str): stem file name. Actual file name will be `save_file_stem`.0, `save_file_stem`.1, etc.
            params_per_file (int): number of params per file,default is 1 billion

        Returns:
            None
        """
        #TODO: use FxModuleParser.ATTR_CONTENT_FILE_FORMAT to name the files.
        total_size = sum([val.numel() for _, (_, val) in self._attr_map.items()])
        model_pt_part_num = (total_size + params_per_file - 1) // params_per_file

        tid2value = {t.tid: val.cpu() for t, (_, val) in self._attr_map.items()}
        # it can be zero if there is no param in the module (self._attr_map is empty)
        if model_pt_part_num <= 1:
            torch.save(tid2value, f'{save_file_stem}.0')
        else:
            tids = list(tid2value.keys())
            assert len(tids) > 0, "Empty attr map"
            chunk_size = (len(tids) + model_pt_part_num - 1) // model_pt_part_num
            chunks = [tids[i:min(i + chunk_size, len(tids))] for i in
                      range(0, len(tids), chunk_size)]
            for idx, chunk in enumerate(chunks):
                assert len(chunk) > 0, f"Empty chunk {idx}"
                part = {k: tid2value[k] for k in chunk}
                torch.save(part, f'{save_file_stem}.{idx}')

    def save_attr_map(self, save_file: str = 'dist_param_map.pt'):
        """
        Save local_param -> origin_param name map.
        """
        ir_name_to_orig_name = {str(t.name).replace('.', '_'): name for t, (name, _) in self._attr_map.items()}
        torch.save(ir_name_to_orig_name, save_file)

    def push_param(self, var_name):
        """
        push var name to the method stack

        Args:
            var_name (str): variable name
        """
        if var_name not in self._vars[-1]:
            raise KeyError(f"push {var_name} not declared")
        self._var_stack.append(var_name)

    def pop_param(self, times=1):
        """
        pop var name from the method stack
        """
        for _ in range(times):
            self._var_stack.pop()

    def depth(self):
        return len(self._vars)

    def __repr__(self):
        dscp = f'frame: depth: {self.depth()}\n  var table:'
        for var_name in self._vars[-1].keys():
            dscp += f'\n    {var_name} : {self._vars[-1][var_name]}'
        dscp += f'\n  var stack:'
        for var_name in self._var_stack:
            dscp += f'\n    {var_name}'
        return dscp
