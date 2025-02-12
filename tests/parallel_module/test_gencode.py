#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import inspect
import tempfile
import re
from contextlib import nullcontext
from typing import Union

import torch
import torch.nn.functional as F
import pytest
from unittest.mock import patch

from nnscaler.flags import CompileFlag
import nnscaler.graph.function.dimops
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.parser.mapping import SignFx2Op
from nnscaler.ir.cten import IR, IRObject
from nnscaler.parallel import parallelize, ComputeConfig, CubeModule, _gen_graph

from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import replace_all_device_with, raises_with_cause

def _to_cube_model(module, compute_config, cube_savedir, load_module):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'data',
        compute_config,
        gen_savedir=cube_savedir,
        load_module=load_module
    )

class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


def _gencode_worker(tempdir):
    init_distributed()
    m = Module0()
    with pytest.raises(RuntimeError):  # config mismatch
        pm = _to_cube_model(m, ComputeConfig(1, 1), cube_savedir=tempdir, load_module=True)
        with pytest.raises(NotImplementedError):
            pm._train_step(None)  # for non-end2end parallel module, _train_step is not implemented


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen():
    with tempfile.TemporaryDirectory() as tempdir:
        m = Module0()
        m_new = _to_cube_model(m, ComputeConfig(2, 4), cube_savedir=tempdir, load_module=False)
        assert m_new is None
        launch_torchrun(1, _gencode_worker, tempdir)


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:2]

@replace_all_device_with('cpu')
def test_codegen_slice():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            SliceModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        assert m_new is None


class ArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, *args):
        return self.linear(x) + y


@replace_all_device_with('cpu')
def test_codegen_args():
    with tempfile.TemporaryDirectory() as tempdir:
        # *args is not supported.
        with pytest.raises(RuntimeError):
            parallelize(
                ArgsModule(),
                {
                    'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    'y': 1.0,
                },
                'dp',
                ComputeConfig(1, 1),
                gen_savedir=tempdir,
                load_module=True
            )


class TupleReturnModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y):
        return self.linear(x) + y, y + 10


@replace_all_device_with('cpu')
def test_codegen_tuple_return1():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            TupleReturnModule1(),
            {
                'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                'y': 1.0,
            },
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        assert len(_gencode_contains(tempdir, TupleReturnModule1, 0,
            r"return add_.*, add_.*")) == 2


class TupleReturnModule2(torch.nn.Module):
    def __init__(self, return_type):
        super().__init__()
        self.return_type = return_type
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y):
        if self.return_type == 0:
            return self.linear(x),
        else:
            return [[self.linear(x) + y]]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('return_type', [0, 1])
def test_codegen_tuple_return2(return_type):
    test_context = nullcontext() if return_type != 0 else raises_with_cause(RuntimeError, match='Single tuple outputs.*')
    with tempfile.TemporaryDirectory() as tempdir, test_context:
        parallelize(
            TupleReturnModule2(return_type),
            {
                'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                'y': 1.0,
            },
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        assert _gencode_contains(tempdir, TupleReturnModule2, 0,
            r"return \[\[add_.*\]\]")


class UnusedArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, z=None, m=1, n=2, **kwargs):
        return self.linear(x) + m


def _gencode_unused_args_worker(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgsModule(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'z': None,
            'm': 0,
            'n': None,
         },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 6
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['z'].default is None
    assert args.parameters['m'].default == 1
    assert args.parameters['n'].default is None
    assert args.parameters['kwargs'].default is inspect.Parameter.empty

    assert torch.equal(
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    )

    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_unused_args():
    """
    Verify that unused args are supported by parallalize
    """
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker, tempdir)


class UnusedArgs2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, m):
        return self.linear(x) + m


def _gencode_unused_args_worker2(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgs2Module(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'm': 0
         },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 3
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['m'].default is None

    m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    with pytest.raises(TypeError, match='.*must be Tensor, not NoneType.*'):
        # raise by torch.add, as m is None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_unused_args2():
    """
    Verify that unused args are supported by parallalize
    """
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker2, tempdir)


def pas_dp_with_recompute(graph, cfg):
    """
    pure data parallelism policy
    """
    from nnscaler.ir import IRFwOperation, IRDataOperation
    from nnscaler.policies import _replica
    ngpus = cfg.plan_ngpus
    if ngpus != 1:
        raise ValueError("Data parallelism only supports 1 plan GPU")

    # combine
    # x = _add(x, v1)
    # x = x + v2
    # x = self.linear(x)
    # together as a recompute unit
    graph.recompute([
        *graph.select(name='_add', ntype=IRFwOperation),
        *graph.select(name='add', ntype=IRFwOperation),
        *graph.select(name='linear', ntype=IRFwOperation),
    ])
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        _replica(graph, node, [0])
    return graph


@nnscaler.register_op('* -> *')
def _add(x, k):
    return x + k


class RecomputeKwArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, v1, v2):
        x = _add(x, v1)  # v1 will be kwargs
        x = x + v2       # v2 will be normal args
        x = self.linear(x)
        x = x - v2
        return x


@replace_all_device_with('cpu')
def test_codegen_recompute_kwargs():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            RecomputeKwArgsModule(),
            {
                'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                'v1': 1.0,
                'v2': 2.0,
            },
            pas_dp_with_recompute,
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # It will look like
        # def segment33(self, x_36, v1_38, v2_39):
        #     def recompute(x_36, v1_38, v2_39):
        #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 237, in forward,  x = _add(x, v1)  # v1 will be kwargs
        #         _add_29 = tests.parallel_module.test_gencode._add(x_36, k=v1_38)
        #         del x_36
        #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 238, in forward,  x = x + v2       # v2 will be normal args
        #         add_30 = torch.add(_add_29, v2_39, alpha=1)
        #         del _add_29
        #         # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 239, in forward,  x = self.linear(x)
        #         linear_33 = torch.nn.functional.linear(add_30, self.linear_weight_31, self.linear_bias_32)
        #         del add_30
        #         return linear_33
        #     linear_33 = ckpt.checkpoint(recompute, x_36, v1_38, v2_39, use_reentrant=False)
        #     del x_36
        #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 240, in forward,  x = x - v2
        #     sub_28 = torch.sub(linear_33, v2_39, alpha=1)
        #     del linear_33
        #     return sub_28
        assert _gencode_contains(tempdir, RecomputeKwArgsModule, 0,
                r'def recompute\(x_\d+, v1_\d+, v2_\d+\)'
        )



class DefaultArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, m=0, n=None):
        return self.linear(x) + m


@replace_all_device_with('cpu')
def test_codegen_default_args():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            DefaultArgsModule(),
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # parallelize will succeed.
        assert True


class AttrModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attr):
        return x + getattr(attr, 'a')


def print_gencode(cubesave_dir, module_class, index=0):
    from nnscaler.parallel import _PARALLEL_MODULE_NAMESPACE, _get_full_qualified_name, _DEFAULT_INSTANCE_NAME
    from pathlib import Path
    import re
    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}.{_DEFAULT_INSTANCE_NAME}'
    outdir: Path = cubesave_dir / Path(namespace.replace('.', '/').strip('/'))
    filecontent = (outdir /f'gencode{index}.py').read_text()
    print(filecontent)


def _gencode_contains(cubesave_dir, module_class, index, search_re):
    from nnscaler.parallel import _PARALLEL_MODULE_NAMESPACE, _get_full_qualified_name, _DEFAULT_INSTANCE_NAME
    from pathlib import Path
    import re
    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}.{_DEFAULT_INSTANCE_NAME}'
    outdir: Path = cubesave_dir / Path(namespace.replace('.', '/').strip('/'))
    filecontent = (outdir /f'gencode{index}.py').read_text()
    matches = re.findall(search_re, filecontent)
    return matches


class AttrHelper:
    def __init__(self) -> None:
        self.a = 2.0


@replace_all_device_with('cpu')
def test_codegen_attr():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            AttrModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0]), 'attr': AttrHelper()},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # in old version, all 'forward' functions will patched to a function named 'new_func'
        assert not _gencode_contains(tempdir, AttrModule, 0, r'new_func')
        assert _gencode_contains(tempdir, AttrModule, 0, r'builtins.getattr\(.*, \'a\'\)')
        assert m_new is None


class GetItemModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data):
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        return padding_mask


@replace_all_device_with('cpu')
def test_codegen_getitem():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            GetItemModule(),
            {'batched_data': {'x': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]])}},
            'tp',
            ComputeConfig(2, 2),
            gen_savedir=tempdir,
            load_module=False,
        )
        assert _gencode_contains(tempdir, GetItemModule, 0, r'_operator.getitem\(.*, slice\(None, 2, None\)\)')
        assert _gencode_contains(tempdir, GetItemModule, 1, r'_operator.getitem\(.*, slice\(None, 2, None\)\)')
        assert m_new is None


class TrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        if self.training:
            return self.linear(x)
        else:
            return self.linear(x) + 1


@replace_all_device_with('cpu')
def test_codegen_training_flag():
    with tempfile.TemporaryDirectory() as tempdir:
        m = TrainingModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )


# class IdentityModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x


# def test_codegen_identity():
#     """
#     Test it can support modules without parameters
#     """
#     if not torch.cuda.is_available():
#         print('skip test_codegen_iter due to lack of cuda devices')
#         return
#     with tempfile.TemporaryDirectory() as tempdir:
#         m = IdentityModule()
#         m.train()
#         parallelize(
#             m,
#             {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
#             PASData,
#             ComputeConfig(1, 2),
#             cube_savedir=tempdir,
#             load_module=False
#         )
#         assert False


class IterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        assert list(x.shape) == [2, 5]  # will generate iter here.
        return x


@replace_all_device_with('cpu')
def test_codegen_iter():
    """
    Test it can support modules without parameters
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = IterModule()
        m.train()
        # assert no exception raised below
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )


class ConstantModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        y = int(x.shape[-1])
        x = x[:, :y]
        return x


@replace_all_device_with('cpu')
def test_codegen_const():
    """
    Test it can support modules without parameters
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = ConstantModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        assert not _gencode_contains(tempdir, ConstantModule, 0, r'\s+5 = builtins.int')


class TensorSliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        padding = torch.count_nonzero(x)
        return x[:, :padding]


class TensorSliceFixedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        padding = torch.count_nonzero(x).item()
        return x[:, :padding]


@replace_all_device_with('cpu')
def test_codegen_tensor_slice():
    with tempfile.TemporaryDirectory() as tempdir:
        m = TensorSliceModule()
        m.train()
        with raises_with_cause(RuntimeError, match='Tensor is not supported in slice.'):
            parallelize(
                m,
                {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
                'dp',
                ComputeConfig(1, 1),
                gen_savedir=tempdir,
                load_module=False,
                reuse='override',
            )
        m = TensorSliceFixedModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False,
            reuse='override',
        )


class DictGetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data: dict):
        data_x = batched_data["x"]
        data_y = batched_data.get("y", batched_data['z'])
        return data_x + data_y


@replace_all_device_with('cpu')
def test_codegen_dictget():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            DictGetModule(),
            {'batched_data': {
                'x': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]]),
                'z': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]])
            }},
            'tp',
            ComputeConfig(2, 2),
            gen_savedir=tempdir,
            load_module=False,
        )
        assert _gencode_contains(tempdir, DictGetModule, 0, r"dict.get\(\w+, 'y', \w+\)")
        assert _gencode_contains(tempdir, DictGetModule, 1, r"dict.get\(\w+, 'y', \w+\)")
        assert m_new is None


class CloneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


@replace_all_device_with('cpu')
def test_codegen_clone():
    with tempfile.TemporaryDirectory() as tempdir:
        g, _ = _gen_graph(
            CloneModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            tempdir,
            True
        )
        assert isinstance(g.nodes()[0], nnscaler.graph.function.dimops.IRDimops)


class MinModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.min(a, b)

def _gencode_min_function_worker(tempdir):
    init_distributed()
    m_new = parallelize(
        MinModule(),
        {
            'a': torch.tensor([5, 2, 3]),
            'b': torch.tensor([1, 8, 1]),
        },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 2
    assert args.parameters['a'].default is inspect.Parameter.empty
    assert args.parameters['b'].default is inspect.Parameter.empty


    assert torch.equal(m_new(torch.tensor([5, 2, 3]), torch.tensor([1, 8, 1])), torch.tensor([1, 2, 1])), "Expected element-wise min"
    assert torch.equal(m_new(torch.tensor([-5, -2, -3]), torch.tensor([-1, -8, -1])), torch.tensor([-5, -8, -3])), "Expected element-wise min with negative values"


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_min():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_min_function_worker, tempdir)


class MaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        return torch.max(a, dim=1, keepdim=True)[0]

def _gencode_max_function(tempdir):
    init_distributed()
    m_new = parallelize(
        MaxModule(),
        {
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl).parameters

    assert len(args) == 1, "Expected 1 argument in the forward method"
    assert args['a'].default is inspect.Parameter.empty, "Expected 'a' to have no default value"

    expected_output = torch.tensor([[3], [6]])
    actual_output = m_new(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(actual_output, expected_output), "Expected each row's max value with original dimension"


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of GPU devices')
def test_codegen_max():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_max_function, tempdir)


class SharedParameterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear2.weight = self.linear1.weight  # shared parameter

    def forward(self, x):
        return self.linear2(self.linear1(x))


@replace_all_device_with('cpu')
def test_codegen_shared_parameter():
    with tempfile.TemporaryDirectory() as tempdir:
        m = SharedParameterModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False,
            reuse='override',
        )
        assert _gencode_contains(tempdir, SharedParameterModule, 0, r"self\.register_parameter\('linear1_bias_*")
        assert _gencode_contains(tempdir, SharedParameterModule, 0, r"self\.register_parameter\('linear2_bias_*")
        assert _gencode_contains(tempdir, SharedParameterModule, 0, r"self\.register_parameter\('linear1_weight_*")
        # linear2_weight shares the same parameter with linear1_weight
        # so there will be no linear2_weight in the generated code
        assert not _gencode_contains(tempdir, SharedParameterModule, 0, r"self\.register_parameter\('linear2_weight_*")


class BufferModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer', torch.ones(128, 64), persistent=False)
        self.fc = torch.nn.Linear(64, 64)

    # x with shape [128, 64]
    def forward(self, x):
        return self.fc(x + self.buffer)


@replace_all_device_with('cpu')
def test_codegen_buffer():
    """
    Test even the buffer is not persistent,
    it will be registered in the generated code as a persistent buffer.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = BufferModule()
        m.train()
        parallelize(
            m,
            {'x': torch.randn(128, 64)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False,
            reuse='override',
        )
        matches =  _gencode_contains(tempdir, BufferModule, 0,
            r"self\.register_buffer\('buffer_*"
        )
        assert len(matches) == 1
        match = matches[0]
        assert 'persistent' not in match


class End2EndModule(torch.nn.Module):
    def __init__(self, dim: int = 1024, nlayers: int = 16):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data: torch.Tensor, return_type: int = 0):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        if return_type == 0:
            return loss
        elif return_type == 1:
            return loss, data.shape # the second return is not tensor
        elif return_type == 2:
            return loss, {'data': data}
        elif return_type == 3:
            return torch.sum(x, -1)  # bad loss
        elif return_type == 4:
            return {'data': data}  # not tensor


@replace_all_device_with('cpu')
def test_codegen_inference():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            Module0(),
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            'dp',
            ComputeConfig(1, 1, inference_only=True),
            gen_savedir=tempdir,
            load_module=False
        )
        assert _gencode_contains(tempdir, Module0, 0,
                r"self\.register_buffer"
        )
        assert not _gencode_contains(tempdir, Module0, 0,
                r"self\.register_parameter"
        )


@replace_all_device_with('cpu')
def test_codegen_end2end():
    """
    Test end2end code generation for different configs
    (use_pipeline, dynamic shape, return value)
    """
    dim = 1024
    nlayers = 16
    batch_size = 64
    def p(cube_dir, use_pipeline, constant_folding, return_type, inference_only=False):
        m = End2EndModule(dim, nlayers)
        m.train()
        parallelize(
            m,
            {'data': torch.randn(batch_size, dim), 'return_type': return_type},
            'data' if not use_pipeline else 'pp',
            compute_config= ComputeConfig(
                4, 4,
                inference_only=inference_only,
                constant_folding=constant_folding,
                use_end2end=True,
                pas_config=dict(
                    pipeline_nmicros=4,
                    pipeline_nstages=4,
                    pipeline_scheduler='infer_pipe' if inference_only else '1f1b'
                )
            ),
            gen_savedir=cube_dir,
            load_module=False,
            reuse='override',
        )
    with tempfile.TemporaryDirectory() as tempdir:
        for use_pipeline in [True, False]:
            p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=0) # should success
            assert not _gencode_contains(tempdir, End2EndModule, 0,
                    r"self\.register_buffer"
            )
            assert _gencode_contains(tempdir, End2EndModule, 0,
                    r"self\.register_parameter"
            )
            p(tempdir, use_pipeline=use_pipeline, constant_folding=True, return_type=0)  # should success
            if use_pipeline:
                with raises_with_cause(RuntimeError, match='.*Communication generation.*'):
                    # fail for non-tensor IRObject return in pipeline mode
                    p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=1)
            else:
                p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=1)
            p(tempdir, use_pipeline=use_pipeline, constant_folding=True, return_type=1)  # should success
            p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=2)  # should success
            p(tempdir, use_pipeline=use_pipeline, constant_folding=True, return_type=2)  # should success
            with raises_with_cause(RuntimeError, match='.*Loss can only be scalar tensor.*'):
                p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=3)
            with raises_with_cause(RuntimeError, match='.*Loss can only be scalar tensor.*'):
                p(tempdir, use_pipeline=use_pipeline, constant_folding=True, return_type=3)
            with raises_with_cause(RuntimeError, match='.*Loss can only be scalar tensor.*'):
                p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=4)
            with raises_with_cause(RuntimeError, match='.*Loss can only be scalar tensor.*'):
                p(tempdir, use_pipeline=use_pipeline, constant_folding=True, return_type=4)

            p(tempdir, use_pipeline=use_pipeline, constant_folding=False, return_type=0, inference_only=True)  # should success
            assert not _gencode_contains(tempdir, End2EndModule, 0,
                    r"self\.register_parameter"
            )
            assert _gencode_contains(tempdir, End2EndModule, 0,
                    r"self\.register_buffer"
            )

from dataclasses import dataclass
@dataclass
class DataT:
    x: int = 0
    y: int = 0


class DropoutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._data = DataT()

    def forward(self, x):
        x = x + self._data.x
        return torch.nn.functional.dropout(x, 0.1 if self.training else 0.2, self.training)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('constant_fold', [False, True])
def test_codegen_dropout(constant_fold):
    """
    Test if self.training is correctly handled in the generated code
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = DropoutModule()
        m.train()
        parallelize(
            m,
            {'x': torch.randn(128, 64)},
            'dp',
            ComputeConfig(1, 1, constant_folding=constant_fold),
            gen_savedir=tempdir,
            load_module=False,
            reuse='override',
        )
        # it should looks like:
        # add_17 = torch.add(x_20, 0, alpha=1)
        # del x_20
        # training_9 = self.training
        # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 778, in forward,  return torch.nn.functional.dropout(x, 0.1 if self.training else 0.2, self.training)
        # ifexpr_4 = 0.1 if training_9 else 0.2
        # training_1_12 = self.training
        # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 778, in forward,  return torch.nn.functional.dropout(x, 0.1 if self.training else 0.2, self.training)
        # dropout_16 = torch.nn.functional.dropout(add_17, p=ifexpr_4, training=training_1_12, inplace=False)
        # del add_17
        # return dropout_16
        assert _gencode_contains(tempdir, DropoutModule, 0,
                r"ifexpr_\d+ = 0.1 if training_\d+ else 0.2"
        )
        assert _gencode_contains(tempdir, DropoutModule, 0,
                    r" = torch.nn.functional.dropout\(add_\d+, p=ifexpr_\d+, training=training_1_\d+, inplace=False\)"
            )


@nnscaler.register_op('?, ? -> ?')
def get_dropout(training, dropout):
    return dropout if training else 0.0


class DropoutModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.dropout(x, get_dropout(self.training, 0.2), self.training)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('constant_fold', [False, True])
def test_codegen_dropout2(tmp_path, constant_fold):
    """
    Test if register_op is correctly handled in the generated code
    """
    m = DropoutModule2()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1, constant_folding=constant_fold),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # it should looks like:
    # training_7 = self.training
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training), self.training)
    # get_dropout_3 = tests.parallel_module.test_gencode.get_dropout(training_7)
    # training_1_11 = self.training
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training), self.training)
    # dropout_15 = torch.nn.functional.dropout(x_18, p=get_dropout_3, training=training_1_11, inplace=False)
    # del x_18
    # return dropout_15
    assert _gencode_contains(tmp_path, DropoutModule2, 0,
            r"= tests.parallel_module.test_gencode.get_dropout\(training_\d+"
    )
    assert _gencode_contains(tmp_path, DropoutModule2, 0,
            r"= torch.nn.functional.dropout\(x_\d+, p=get_dropout_\d+"
    )


class DropoutModuleNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = DropoutModule2()

    def forward(self, x):
        return self.dropout(x)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('constant_fold', [False, True])
def test_codegen_dropout_nested(tmp_path, constant_fold):
    """
    Test if register_op is correctly handled in the generated code
    """
    m = DropoutModuleNested()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1, constant_folding=constant_fold),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # it should looks like:
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training, 0.2), self.training)
    # dropout_training_7 = self.training
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training, 0.2), self.training)
    # get_dropout_3 = tests.parallel_module.test_gencode.get_dropout(dropout_training_7, 0.2)
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training, 0.2), self.training)
    # dropout_training_1_11 = self.training
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 838, in forward,  return torch.nn.functional.dropout(x, get_dropout(self.training, 0.2), self.training)
    # dropout_15 = torch.nn.functional.dropout(x_18, p=get_dropout_3, training=dropout_training_1_11, inplace=False)
    # del x_18
    # return dropout_15
    assert _gencode_contains(tmp_path, DropoutModuleNested, 0,
            r"= tests.parallel_module.test_gencode.get_dropout\(dropout_training_\d+"
    )
    assert _gencode_contains(tmp_path, DropoutModuleNested, 0,
            r"= torch.nn.functional.dropout\(x_\d+, p=get_dropout_\d+"
    )


class DictOutputModule(torch.nn.Module):
     def forward(self, x):
        return {'data': x + 10}


@replace_all_device_with('cpu')
def test_codegen_dictout(tmp_path):
    m = DictOutputModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # it should looks like:
    # def segment9(self, x_9):
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 819, in forward,  return {'data': x + 10}
    #     add_6 = torch.add(x_9, 10, alpha=1)                                                                                                             del x_9
    #     return add_6

    # def _forward_impl(self, x):                                                                                                                         add_6 = self.segment9(x)
    #     return {'data': add_6}
    assert _gencode_contains(tmp_path, DictOutputModule, 0,
            r"return {'data': add_\d+}"
    )


class ReduceScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 1024, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


def pas_reduce_scatter(graph, cfg):
    from nnscaler.ir import IRFwOperation, IRDataOperation
    from nnscaler.policies import _tp, _replica
    ngpus = cfg.plan_ngpus

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'linear':
            _tp(graph, node, list(range(ngpus)), 0, 1)
        elif node.name == 'relu':
            _tp(graph, node, list(range(ngpus)), 0, 0)
        else:
            _replica(graph, node, list(range(ngpus)))
    return graph


@replace_all_device_with('cpu')
@pytest.mark.parametrize('disable_reduce_scatter_adapter', [True, False])
def test_codegen_reduce_scatter(tmp_path, disable_reduce_scatter_adapter):
    old = CompileFlag.disable_reduce_scatter_adapter
    CompileFlag.disable_reduce_scatter_adapter = disable_reduce_scatter_adapter
    m = ReduceScatterModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(2, 512)},
        pas_reduce_scatter,
        ComputeConfig(2, 2),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # With reduce-scatter, it should looks like:
    # ...
    # linear_40 = nnscaler.runtime.adapter.nn.reducescatter_allgather(linear_30, dim=0, ranks=[0, 1])
    # ...

    # without reduce-scatter, it should looks like:
    # ...
    # class Adapter24(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, linear_30):
    #         linear_18 = nnscaler.runtime.adapter.all_reduce(linear_30, ranks=[0, 1])
    #         linear_40 = nnscaler.runtime.adapter.chunk(linear_18, dim=0, ranks=[0, 1])
    #         return linear_40
    #     @staticmethod
    #     def backward(ctx, glinear_48):
    #         glinear_25 = nnscaler.runtime.adapter.all_gather(glinear_48, dim=0, ranks=[0, 1])
    #         return glinear_25
    # ...
    CompileFlag.disable_reduce_scatter_adapter = old
    if not disable_reduce_scatter_adapter:
        assert _gencode_contains(tmp_path, ReduceScatterModule, 0,
                r"nnscaler.runtime.adapter.nn.reducescatter_allgather"
        )
    else:
        assert not _gencode_contains(tmp_path, ReduceScatterModule, 0,
                r"nnscaler.runtime.adapter.nn.reducescatter_allgather"
        )


class CVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channel = 32
        self.kernel_size = 3

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        input = input.view(1, batch * in_channel, height, width)
        weight = torch.randn(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        return out


def pas_conv2d(graph, cfg):
    from nnscaler.ir import IRFwOperation, IRDataOperation
    from nnscaler.policies import _tp, _replica
    ngpus = cfg.plan_ngpus

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'conv_transpose2d':
            # this is an invalid partition
            # ValueError will be raised
            _tp(graph, node, list(range(ngpus)), 1, 1)
        else:
            _replica(graph, node, list(range(ngpus)))
    return graph


@replace_all_device_with('cpu')
def test_invalid_partition(tmp_path):
    """
    ConvTranspose2D and ConvTranspose1D oC dim can't be split
    """
    batch, in_channel, height, width = 2, 16, 32, 32
    input = torch.randn((batch, in_channel, height, width))

    dummy_input = {'input': input}

    m = CVModel()
    m.train()

    with raises_with_cause(ValueError):
        parallelize(
            m,
            dummy_input,
            pas_conv2d,
            ComputeConfig(2, 2),
            gen_savedir=tmp_path,
            load_module=False,
            reuse='override',
        )


class KwargsModule(torch.nn.Module):
     def forward(self, x):
        return x + torch.zeros_like(x, dtype=torch.float32)


@replace_all_device_with('cpu')
def test_codegen_kwargs(tmp_path):
    m = KwargsModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # it should looks like:
    # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 861, in forward,  return x + torch.zeros_like(x, dtype=torch.float32)
    # zeros_like_9 = torch.zeros_like(x_12, requires_grad=False, dtype=torch.float32)
    # # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 861, in forward,  return x + torch.zeros_like(x, dtype=torch.float32)
    # add_8 = torch.add(x_12, zeros_like_9, alpha=1)
    # del x_12, zeros_like_9
    # return add_8
    assert _gencode_contains(tmp_path, KwargsModule, 0,
            r"torch.zeros_like\(x_\d+, requires_grad=False, dtype=torch.float32\)"
    )


class ScalarTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1024, 1024, bias=False)
        self.scale = torch.nn.Parameter(torch.zeros(64))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.num_batches_tracked: torch.Tensor

    def forward(self, x):
        self.num_batches_tracked.add_(1)
        x = self.proj(x)
        coef = torch.exp(torch.sum(self.scale, dim=-1))
        x = x / coef
        return x.sum()


@replace_all_device_with('cpu')
def test_codegen_scalar_tensor(tmp_path):
    m = ScalarTensorModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(1024, 1024)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # the code will look like this:
    # def __init__(self, init_params=True):
    #     super().__init__()
    #     # communication groups

    #     self.register_buffer('num_batches_tracked_33', torch.empty((), dtype=torch.int64), persistent=True)
    #     self.add_full_map('num_batches_tracked_33', 2, False, 'num_batches_tracked', (), ..., 1)

    #     self.register_parameter('proj_weight_35', torch.nn.Parameter(torch.empty((1024, 1024), dtype=torch.float32)))
    #     self.add_full_map('proj_weight_35', 4, True, 'proj.weight', (1024, 1024), (slice(0, 1024, None), slice(0, 1024, None)), 1)

    #     self.register_parameter('scale_37', torch.nn.Parameter(torch.empty((64,), dtype=torch.float32)))
    #     self.add_full_map('scale_37', 8, True, 'scale', (64,), (slice(0, 64, None),), 1)


    #     self._post_init(init_params)

    # def segment41(self, x_43):
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 990, in forward,  self.num_batches_tracked.add_(1)
    #     add__34 = torch.Tensor.add_(self.num_batches_tracked_33, 1)
    #     del add__34
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 991, in forward,  x = self.proj(x)
    #     linear_36 = torch.nn.functional.linear(x_43, self.proj_weight_35, bias=None)
    #     del x_43
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 992, in forward,  coef = torch.exp(torch.sum(self.scale, dim=-1))
    #     sum_1_38 = torch.sum(self.scale_37, dim=(-1,), keepdim=False)
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 992, in forward,  coef = torch.exp(torch.sum(self.scale, dim=-1))
    #     exp_39 = torch.exp(sum_1_38)
    #     del sum_1_38
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 993, in forward,  x = x / coef
    #     truediv_40 = torch.div(linear_36, exp_39, rounding_mode=None)
    #     del linear_36, exp_39
    #     # File "/home/weijiangxu/MagicCube/tests/parallel_module/test_gencode.py", line 994, in forward,  return x.sum()
    #     sum_2_32 = torch.sum(truediv_40)
    #     del truediv_40
    #     return sum_2_32

    assert _gencode_contains(tmp_path, ScalarTensorModule, 0,
        r"self\.register_buffer\('num_batches_tracked_\d+', torch\.empty\(\(\), dtype=torch\.int64\), persistent=True\)")
    assert _gencode_contains(tmp_path, ScalarTensorModule, 0,
        r"self\.add_full_map\('num_batches_tracked_\d+', \d+, False, 'num_batches_tracked', \(\), \.\.\., 1\)")


class ImportlibModel(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.model = torch.nn.Linear(1024, 1024)

   def forward(self, data):
       import importlib
       x = importlib.import_module('datetime')
       r = self.model(data + x.datetime.now().year)
       return torch.sum(r)


@replace_all_device_with('cpu')
def test_codegen_importlib(tmp_path):
    m = ImportlibModel()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(1024, 1024)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    # should success
    assert True


class ImportlibModel2(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.model = torch.nn.Linear(1024, 1024)

   def forward(self, data):
       torch._dynamo
       r = self.model(data)
       return torch.sum(r)


@replace_all_device_with('cpu')
@pytest.mark.skipif(torch.torch_version.__version__ < (2,1,0), reason='torch._dynamo is not a valid import')
def test_codegen_importlib2(tmp_path):
    m = ImportlibModel2()
    m.train()
    parallelize(
        m,
        {'data': torch.randn(1024, 1024)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    import  nnscaler.graph.tracer.orig_func as orig_func
    import importlib
    assert orig_func.import_module == importlib.import_module
    # should success
    assert True


class ConvTranspose1DModule(torch.nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input, **kwargs):
        groups = kwargs.get('groups', self.groups)
        return torch.nn.functional.conv_transpose1d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, groups, self.dilation)


def _gencode_conv_transpose1d_function(tempdir):
    init_distributed()
    weight = torch.randn(3, 3, 3)
    bias = torch.randn(3)
    m_new = parallelize(
        ConvTranspose1DModule(weight, bias),
        {
            'input': torch.randn(2, 3, 4),
            'groups': 1,
        },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl).parameters
    assert len(args) == 2
    assert args['input'].default is inspect.Parameter.empty, "Expected 'input' to have no default value"
    assert args['kwargs'].default == inspect.Parameter.empty, "Expected 'kwargs' to have no default value"

    input_tensor = torch.randn(2, 3, 4)
    model = ConvTranspose1DModule(weight, bias)
    expected_output = model(input_tensor, groups=1)
    actual_output = m_new(input_tensor, groups=1)
    assert torch.allclose(actual_output, expected_output, atol=1e-6), "Expected the output of ConvTranspose1DModule to match the expected output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of GPU devices')
def test_codegen_conv_transpose1d():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_conv_transpose1d_function, tempdir)


class Conv2DModule(torch.nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input, **kwargs):
        groups = kwargs.get('groups', self.groups)
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, groups)


def _gencode_conv2d_function(tempdir):
    init_distributed()
    weight = torch.randn(3, 3, 3, 3)
    bias = torch.randn(3)
    m_new = parallelize(
        Conv2DModule(weight, bias),
        {
            'input': torch.randn(2, 3, 32, 32),
            'groups': 1,
        },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl).parameters
    assert len(args) == 2
    assert args['input'].default is inspect.Parameter.empty, "Expected 'input' to have no default value"
    assert args['kwargs'].default == inspect.Parameter.empty, "Expected 'kwargs' to have no default value"
    input_tensor = torch.randn(2, 3, 32, 32)
    model = Conv2DModule(weight, bias)
    expected_output = model(input_tensor, groups=1)
    actual_output = m_new(input_tensor, groups=1)
    assert torch.allclose(actual_output, expected_output, atol=1e-6), "Expected the output of Conv2DModule to match the expected output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of GPU devices')
def test_codegen_conv2d():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_conv2d_function, tempdir)


def _gencode_conv2d_function_(tempdir):
    init_distributed()
    weight = torch.randn(6, 3, 3, 3)
    bias = torch.randn(6)
    m_new = parallelize(
        Conv2DModule(weight, bias, groups=2),
        {
            'input': torch.randn(2, 6, 32, 32),
            'groups': 2,
        },
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl).parameters
    assert len(args) == 2
    assert args['input'].default is inspect.Parameter.empty, "Expected 'input' to have no default value"
    assert args['kwargs'].default == inspect.Parameter.empty, "Expected 'kwargs' to have no default value"
    input_tensor = torch.randn(2, 6, 32, 32)
    model = Conv2DModule(weight, bias, groups=2)
    expected_output = model(input_tensor, groups=2)
    actual_output = m_new(input_tensor, groups=2)
    assert torch.allclose(actual_output, expected_output, atol=1e-6), "Expected the output of Conv2DModule to match the expected output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of GPU devices')
def test_codegen_conv2d_groups():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_conv2d_function_, tempdir)


class FunctionToModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x.to(0)).to(torch.float32, copy=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of GPU devices')
def test_codegen_function_to(tmp_path):
    parallelize(
        FunctionToModule(),
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False
    )
    # device argument is removed
    # to_23 = torch.Tensor.to(x_29)
    assert _gencode_contains(tmp_path, FunctionToModule, 0, r'to_\d+ = torch\.Tensor\.to\(x_\d+\)')
    # to_1_22 = torch.Tensor.to(linear_26, copy=True, dtype=torch.float32)
    assert _gencode_contains(tmp_path, FunctionToModule, 0, r'torch\.Tensor\.to([^, ]*, copy=True, dtype=torch.float32)')


class CCFModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear3 = torch.nn.Linear(3, 3)
        self.linear4 = torch.nn.Linear(3, 3)
        self.linear5 = torch.nn.Linear(3, 3)
        self.linear6 = torch.nn.Linear(3, 3)
        self.linear7 = torch.nn.Linear(3, 3)
        self.linear8 = torch.nn.Linear(3, 3)

    def forward(self, a: torch.Tensor):
        ashape = a.shape[0]  # not folded
        b = self.linear1(a) + ashape
        bshape = b.shape[0]  # not folded
        with nnscaler.constant_folding():
            d = self.linear3(b) + ashape
            dshape = d.shape[0]  # folded
            e = self.linear4(d) + dshape + bshape + ashape
            with nnscaler.no_constant_folding():
                f = self.linear5(e) + dshape + bshape + ashape
                fshape = f.shape[0]  # not folded
                g = self.linear6(f) + fshape + dshape + bshape + ashape
            gshape = g.shape[0]  # folded
            h = self.linear7(g) + gshape + fshape + dshape + bshape + ashape
        hshape = h.shape[0]
        i = self.linear8(h) + hshape + gshape + fshape + dshape + bshape + ashape
        return i


@replace_all_device_with('cpu')
def test_constant_folding_context(tmp_path):
    parallelize(
        CCFModule(),
        {'a': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'dp',
        ComputeConfig(1, 1, constant_folding=False),
        gen_savedir=tmp_path,
        load_module=False
    )
    # Just check all torch.add code
    add_codes = _gencode_contains(tmp_path, CCFModule, 0, r'.*torch\.add.*')
    assert len(add_codes) == 23

    not_folded_names = ['ashape', 'bshape', 'fshape', 'hshape']
    folded_names = ['dshape', 'gshape']

    def check_op(*names):
        for name in names:
            code = add_codes.pop(0)
            if name in not_folded_names:
                assert re.match(r'\s*add_.* = torch\.add\((linear|add)_.*, getitem_.*, alpha=1\)', code)
            else:
                assert re.match(r'\s*add_.* = torch\.add\((linear|add)_.*, 2, alpha=1\)', code)

    # b = self.linear1(a) + ashape
    check_op('ashape')
    # d = self.linear3(b) + ashape
    check_op('ashape')
    # e = self.linear4(d) + dshape + bshape + ashape
    check_op('dshape', 'bshape', 'ashape')
    # f = self.linear5(e) + dshape + bshape + ashape
    check_op('dshape', 'bshape', 'ashape')
    # g = self.linear6(f) + fshape + dshape + bshape + ashape
    check_op('fshape', 'dshape', 'bshape', 'ashape')
    # h = self.linear7(g) + gshape + fshape + dshape + bshape + ashape
    check_op('gshape', 'fshape', 'dshape', 'bshape', 'ashape')
    # i = self.linear8(h) + hshape + gshape + fshape + dshape + bshape + ashape
    check_op('hshape', 'gshape', 'fshape', 'dshape', 'bshape', 'ashape')

    assert not add_codes


class CCFModule2(torch.nn.Module):
    def __init__(self, fold_input=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.fold_input = fold_input

    def forward(self, a: torch.Tensor):
        from nnscaler.runtime.function import fold_constant
        ashape = a.shape[0]  # not folded
        ashape2 = a.shape[1] # not folded
        ashape3 = ashape + ashape2 # not folded
        with nnscaler.constant_folding():
            if self.fold_input:
                ashape = fold_constant(ashape)
            b = self.linear1(a) + ashape
            if self.fold_input:
                # check if the constant folding is correctly applied to tuple
                # here we have 3 constants to fold
                # In graph, it will be two nodes `fold_constant` and `getitem`
                ashape, ashape2, ashape3 = fold_constant((ashape, ashape2, ashape3))
            b  = b * ashape * ashape2 * ashape3
        return b


@replace_all_device_with('cpu')
@pytest.mark.parametrize('fold_input', [False, True])
def test_fold_constant(tmp_path, fold_input):
    parallelize(
        CCFModule2(fold_input),
        {'a': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'dp',
        ComputeConfig(1, 1, constant_folding=False),
        gen_savedir=tmp_path,
        reuse='override',
        load_module=False
    )
    if fold_input:
        # add_28 = torch.add(linear_31, 2, alpha=1)
        assert _gencode_contains(tmp_path, CCFModule2, 0,
                                 r'add_.* = torch\.add\(linear_.*, 2, alpha=1\)')
        # b  = b * ashape3
        # mul_2_59 = torch.mul(mul_1_65, 5)
        assert _gencode_contains(tmp_path, CCFModule2, 0,
                                 r'mul_.* = torch\.mul\(mul_.*, 5\)')
    else:
        # add_27 = torch.add(linear_30, getitem_20, alpha=1)
        assert _gencode_contains(tmp_path, CCFModule2, 0,
                                 r'add_.* = torch\.add\(linear_.*, getitem_.*, alpha=1\)')
        # b  = b * ashape3
        # mul_2_51 = torch.mul(mul_1_57, add_38)
        assert _gencode_contains(tmp_path, CCFModule2, 0,
                                 r'mul_.* = torch\.mul\(mul_.*, add_.*\)')


@nnscaler.register_op('? ->')
def _op1(k):
    pass


@nnscaler.register_op('? -> ?')
def _op2(k):
    pass


@nnscaler.register_op(' -> ?')
def _op3():
    return 1


@nnscaler.register_op('? -> ?')
def _op4(k):
    return 1 if k else 0


class IRNoneModule(torch.nn.Module):
    def forward(self, x):
        _op1(2)
        r = _op2(3)
        r = _op3() + _op4(r)
        return x + r


@replace_all_device_with('cpu')
def test_no_return(tmp_path):
    m = IRNoneModule()
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        reuse='override',
        load_module=False,
    )
    # it should looks like:
    # def segment19(self, x_23):
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1707, in forward,  _op1(2)
    #     tests.parallel_module.test_gencode._op1(2)
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1708, in forward,  r = _op2(3)
    #     _op2_4 = tests.parallel_module.test_gencode._op2(3)
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1709, in forward,  r = _op3() + _op4(r)
    #     _op3_5 = tests.parallel_module.test_gencode._op3()
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1709, in forward,  r = _op3() + _op4(r)
    #     _op4_6 = tests.parallel_module.test_gencode._op4(_op2_4)
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1709, in forward,  r = _op3() + _op4(r)
    #     add_15 = _operator.add(_op3_5, _op4_6)
    #     # File "/home/weijiangxu/nanogpt/MagicCube/tests/parallel_module/test_gencode.py", line 1710, in forward,  return x + r
    #     add_1_20 = torch.add(x_23, add_15, alpha=1)
    #     del x_23
    #     return add_1_20

    #  _op1 will not be removed by DCE in tracer
    assert _gencode_contains(tmp_path, IRNoneModule, 0,
                                 r'tests\.parallel_module\.test_gencode\._op1')


class IRUseNoneModule(torch.nn.Module):
    def forward(self, x):
        r = _op3() + _op4(_op1(2))
        return x + r


@replace_all_device_with('cpu')
def test_use_none_return(tmp_path):
    m = IRUseNoneModule()
    m.train()
    # it should raise an error, because _op1 has no return value, but it is used in _op4
    with raises_with_cause(KeyError):
        parallelize(
            m,
            {'x': torch.randn(128, 64)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tmp_path,
            reuse='override',
            load_module=False,
        )



@nnscaler.register_op('? -> ?, ?')
def _op5(k):
    return 1 + k, 2


def _op6(k):
    return 1 + k, 2


# the ops registered with register_op can't cover all code path in parser
def Op6(o: Union[int, IRObject], signature=None):
    o = IR.try_unwrap(o)
    return IRPyFunc(signature, inputs=[o], outputs=[
        IRObject(name='_op6', value=o + 1, is_constant=True),
        IRObject(name='_op6', value=2, is_constant=True),
    ])


class IRMultiOutputModule(torch.nn.Module):
    def forward(self, x):
        r0, _ = _op5(2)
        r1, _ = _op6(3)
        return x + r0 + r1



@replace_all_device_with('cpu')
def test_multi_output_op(tmp_path):
    SignFx2Op.kOpMap['tests.parallel_module.test_gencode._op6'] = Op6

    from nnscaler.graph.tracer import concrete_trace
    from nnscaler.graph.tracer.wrap_utils import LeafWrapInfo
    def patched_concrete_trace(*args, **kwargs):
        kwargs['dce_ignored_function'].add(_op6)
        kwargs['autowrap_leaf_function'][_op6] = LeafWrapInfo([], True, None)

        return concrete_trace(*args, **kwargs)

    with patch(
        "nnscaler.graph.parser.converter.concrete_trace",
        side_effect=patched_concrete_trace
    ):
        m = IRMultiOutputModule()
        m.train()
        parallelize(
            m,
            {'x': torch.randn(128, 64)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tmp_path,
            reuse='override',
            load_module=False,
        )

    SignFx2Op.kOpMap.pop('tests.parallel_module.test_gencode._op6')
    # should success
    assert True


class InitErrorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        raise ValueError('world error')

    def forward(self, input):
        pass


def _gencode_init_error_worker(tmp_path, without_init_distributed=False):
    if not without_init_distributed:
        init_distributed()
    try:
        m_new = parallelize(
            InitErrorModule,
            {
                'input': torch.randn(2, 3, 32, 32),
            },
            'dp',
            ComputeConfig(1, 2),
            gen_savedir=tmp_path,
            load_module=True
        )
    except Exception as e:
        assert isinstance(e, RuntimeError)
        if without_init_distributed or torch.distributed.get_rank() == 0:
            root_cause = e.__cause__
            while root_cause.__cause__ is not None:
                root_cause = root_cause.__cause__
            assert isinstance(root_cause, ValueError)
            assert root_cause.args[0] == 'world error'
        else:
            assert e.__cause__ is None


@replace_all_device_with('cpu')
def test_codegen_init_error_compile(tmp_path):
    _gencode_init_error_worker(tmp_path, without_init_distributed=True)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of GPU devices')
def test_codegen_init__error(tmp_path):
    launch_torchrun(2, _gencode_init_error_worker, tmp_path)


class ForwardErrorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        raise ValueError('hello error')


def _gencode_forward_error_worker(tmp_path, without_init_distributed=False):
    if not without_init_distributed:
        init_distributed()
    try:
        m_new = parallelize(
            ForwardErrorModule,
            {
                'input': torch.randn(2, 3, 32, 32),
            },
            'dp',
            ComputeConfig(1, 2),
            gen_savedir=tmp_path,
            load_module=True
        )
    except Exception as e:
        assert isinstance(e, RuntimeError)
        if without_init_distributed or torch.distributed.get_rank() == 0:
            assert isinstance(e.__cause__, ValueError)
            assert e.__cause__.args[0] == 'hello error'
        else:
            assert e.__cause__ is None


@replace_all_device_with('cpu')
def test_codegen_forward_error_compile(tmp_path):
    _gencode_forward_error_worker(tmp_path, without_init_distributed=True)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of GPU devices')
def test_codegen_forward_error(tmp_path):
    launch_torchrun(2, _gencode_forward_error_worker, tmp_path)
