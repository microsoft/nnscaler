#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
from contextlib import contextmanager
from typing import Any, Callable, List, Dict, NamedTuple

from torch.fx._symbolic_trace import _Patcher
from . import orig_func

class _PatchedFnReusable(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any
    new_fn: Any

    def patch(self):
        raise NotImplementedError()


class _PatchedFnSetItemReusable(_PatchedFnReusable):
    def patch(self):
        self.frame_dict[self.fn_name] = self.new_fn

    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class _PatchedFnDelReusable(_PatchedFnReusable):
    def patch(self):
        self.frame_dict[self.fn_name] = self.new_fn

    def revert(self):
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttrReusable(_PatchedFnReusable):
    def patch(self):
        setattr(self.frame_dict, self.fn_name, self.new_fn)

    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _PatchedFnSetAttrDelReusable(_PatchedFnReusable):
    def patch(self):
        setattr(self.frame_dict, self.fn_name, self.new_fn)

    def revert(self):
        delattr(self.frame_dict, self.fn_name)


class FunctionPatcher(_Patcher):
    def __init__(self):
        super().__init__()
        self.patches_made: List[_PatchedFnReusable] = []
        self.patch_mode = False
        self.in_global_context = False

    def patch(
        self,
        frame_dict: Dict[str, Any],
        name: str,
        new_fn: Callable,
        deduplicate: bool = True,
    ):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        if not self.patch_mode:
            raise RuntimeError('only can do patch in patch mode')
        setattr(new_fn, '__fx_already_patched', deduplicate)
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDelReusable(frame_dict, name, None, new_fn))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItemReusable(frame_dict, name, frame_dict[name], new_fn)
            )
        self.patches_made[-1].patch()

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True, revert_by_del: bool = False
    ):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        if not self.patch_mode:
            raise RuntimeError('only can do patch in patch mode')
        setattr(new_fn, '__fx_already_patched', deduplicate)
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        if revert_by_del:
            self.patches_made.append(_PatchedFnSetAttrDelReusable(cls, name, orig_fn, new_fn))
        else:
            self.patches_made.append(_PatchedFnSetAttrReusable(cls, name, orig_fn, new_fn))
        self.patches_made[-1].patch()

    @contextmanager
    def revert(self):
        if self.in_global_context:
            self._change_patch_mode_to(False)
            for patch in orig_func.reversed(self.patches_made):
                # unpatch in reverse order to handle duplicates correctly
                patch.revert()
            try:
                yield
            finally:
                self._change_patch_mode_to(True)
                for patch in self.patches_made:
                    patch.patch()
        else:
            try:
                yield
            finally:
                pass

    def _change_patch_mode_to(self, to_mode: bool):
        if self.patch_mode != (not to_mode):
            raise RuntimeError(f'want to change patch mode to {to_mode}, but get current patch mode {self.patch_mode}')
        self.patch_mode = to_mode

    def __enter__(self):
        self.in_global_context = True
        self._change_patch_mode_to(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()
        self._change_patch_mode_to(False)
        self.in_global_context = False
