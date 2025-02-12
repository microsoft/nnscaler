#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import ast
import tempfile
import torch

from pathlib import Path
from nnscaler.parallel import parallelize, ComputeConfig, ParallelModule, build_optimizer
from .common import init_distributed, init_random
from .test_end2end import merge_cube_result
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import clear_dir_on_rank0


class CtxManagerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param_1 = torch.nn.Parameter(torch.rand(16, 16))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r_1 = torch.matmul(x, self.param_1)
        r_2 = torch.matmul(y, self.param_1)
        with torch.no_grad():
            r_3 = torch.matmul(r_1, self.param_1)
            with torch.enable_grad():
                r_4 = torch.matmul(r_2, self.param_1)
            with torch.autocast(r_4.device.type):
                r_5 = r_3 * r_4
        r = r_1 * r_2 * r_3 * r_4 * r_5
        return torch.matmul(r, self.param_1).norm()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_ctx_manager_codegen():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, check_ctx_manager_codegen, tempdir)


def dummy_data():
    return {'x': torch.rand(4, 16), 'y': torch.rand(4, 16)}


def check_ctx_manager_codegen(tempdir):
    init_distributed()
    m = CtxManagerModel()
    m_new = parallelize(
        m,
        dummy_data(),
        'data',
        ComputeConfig(2, 4),
        gen_savedir=tempdir,
        load_module=False
    )
    for i in range(4):
        code = get_gencode(tempdir, CtxManagerModel, i)
        ########## Generated Model Code ###########
        # from typing import *
        # from pathlib import Path
        # import torch
        # import torch.utils.checkpoint as ckpt
        # import nnscaler
        # import _operator
        # from numpy import inf
        # import builtins

        # runtime_version = '0.6'


        # import nnscaler.graph.function.wrapnn

        # import apex.normalization.fused_layer_norm

        # class GenModel(nnscaler.runtime.module.ParallelModule):
        #     use_scheduler = False
        #     nmicros_per_scheduler_step = 1
        #     rank = 0
            
        #     def __init__(self, init_params=True, *, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False):
        #         super().__init__()
        #         # communication groups
        #         self.init_group(ranks=[0, 2])
        #         self.init_group(ranks=[1, 3])
        #         self.init_group(ranks=[0, 1])
        #         self.init_group(ranks=[2, 3])
                
        #         self.register_parameter('param_1_62', torch.nn.Parameter(torch.empty((16, 16), dtype=torch.float32)))
        #         self.add_full_map('param_1_62', 5, True, 'param_1', (16, 16), (slice(0, 16, None), slice(0, 16, None)), 1)
                
                
        #         self.wreducer312 = nnscaler.runtime.adapter.Reducer(ranks=[0, 2], reduce_op='sum', async_op=async_op, zero=False, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_ngroups=1)
        #         self.wreducer312.add_param(self.param_1_62)
        #         self.add_reducer(self.wreducer312)
                
        #         self._post_init(init_params)
            
        #     def segment308(self, x_75, y_78):
        #         # auto_multiref
        #         param_1_106, param_1_107, param_1_108, param_1_109, param_1_110 = nnscaler.runtime.function.multiref(self.param_1_62, times=5)
        #         x_166 = nnscaler.runtime.adapter.nn.split_allgather(x_75, dim=0, ranks=[0, 1])
        #         del x_75
        #         param_1_109 = nnscaler.runtime.adapter.nn.identity_allreduce(param_1_109, ranks=[0, 1])
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 21, in forward,  r_1 = torch.matmul(x, self.param_1)
        #         matmul_168 = torch.matmul(x_166, param_1_109)
        #         del param_1_109, x_166
        #         # create at IRAdapterGener:autoref, comment before transformation: auto_multiref
        #         matmul_226, matmul_194 = nnscaler.runtime.function.multiref(matmul_168, times=2)
        #         del matmul_168
        #         y_180 = nnscaler.runtime.adapter.nn.split_allgather(y_78, dim=0, ranks=[0, 1])
        #         del y_78
        #         param_1_110 = nnscaler.runtime.adapter.nn.identity_allreduce(param_1_110, ranks=[0, 1])
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 22, in forward,  r_2 = torch.matmul(y, self.param_1)
        #         matmul_1_182 = torch.matmul(y_180, param_1_110)
        #         del param_1_110, y_180
        #         # create at IRAdapterGener:autoref, comment before transformation: auto_multiref
        #         matmul_1_202, matmul_1_228 = nnscaler.runtime.function.multiref(matmul_1_182, times=2)
        #         del matmul_1_182
                
        #         with torch.no_grad():
        #             # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 24, in forward,  r_3 = torch.matmul(r_1, self.param_1)
        #             matmul_2_196 = torch.matmul(matmul_194, param_1_106)
        #             del param_1_106, matmul_194
                
        #         # create at IRAdapterGener:autoref, comment before transformation: auto_multiref
        #         matmul_2_216, matmul_2_242 = nnscaler.runtime.function.multiref(matmul_2_196, times=2)
        #         del matmul_2_196
        #         param_1_107 = nnscaler.runtime.adapter.nn.identity_allreduce(param_1_107, ranks=[0, 1])
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 26, in forward,  r_4 = torch.matmul(r_2, self.param_1)
        #         matmul_3_204 = torch.matmul(matmul_1_202, param_1_107)
        #         del param_1_107, matmul_1_202
        #         # create at IRAdapterGener:autoref, comment before transformation: auto_multiref
        #         matmul_3_252, matmul_3_218 = nnscaler.runtime.function.multiref(matmul_3_204, times=2)
        #         del matmul_3_204
                
        #         with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True, cache_enabled=True):
        #             # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 28, in forward,  r_5 = r_3 * r_4
        #             mul_220 = torch.mul(matmul_2_216, matmul_3_218)
        #             del matmul_2_216, matmul_3_218
                
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 29, in forward,  r = r_1 * r_2 * r_3 * r_4 * r_5
        #         mul_1_230 = torch.mul(matmul_226, matmul_1_228)
        #         del matmul_226, matmul_1_228
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 29, in forward,  r = r_1 * r_2 * r_3 * r_4 * r_5
        #         mul_2_244 = torch.mul(mul_1_230, matmul_2_242)
        #         del matmul_2_242, mul_1_230
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 29, in forward,  r = r_1 * r_2 * r_3 * r_4 * r_5
        #         mul_3_254 = torch.mul(mul_2_244, matmul_3_252)
        #         del matmul_3_252, mul_2_244
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 29, in forward,  r = r_1 * r_2 * r_3 * r_4 * r_5
        #         mul_4_264 = torch.mul(mul_3_254, mul_220)
        #         del mul_220, mul_3_254
        #         param_1_108 = nnscaler.runtime.adapter.nn.identity_allreduce(param_1_108, ranks=[0, 1])
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 30, in forward,  return torch.matmul(r, self.param_1).norm()
        #         matmul_4_272 = torch.matmul(mul_4_264, param_1_108)
        #         del param_1_108, mul_4_264
        #         matmul_4_72 = nnscaler.runtime.adapter.nn.allgather_split(matmul_4_272, dim=0, ranks=[0, 1])
        #         del matmul_4_272
        #         # File "/scratch/nishang/MagicCube/tests/parallel_module/test_gencode_ctx_manager.py", line 30, in forward,  return torch.matmul(r, self.param_1).norm()
        #         norm_61 = torch.norm(matmul_4_72, p='fro', dim=None, keepdim=False, out=None, dtype=None)
        #         del matmul_4_72
        #         return norm_61
            
        #     def reducer312(self):
        #         self.wreducer312.sync_grads()
        #         return 
            
        #     def _forward_impl(self, x, y):
        #         norm_61 = self.segment308(x, y)
        #         return norm_61

        # with torch.no_grad() as _nnscaler_no_grad:
        def first_with_node_check(node: ast.With):
            hit_no_grad = False
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    func = item.context_expr.func
                    if isinstance(func, ast.Attribute):
                        module_name = func.value.id if isinstance(func.value, ast.Name) else None
                        context_manager_name = func.attr
                        if module_name == 'torch' and context_manager_name == 'no_grad':
                            assert not hit_no_grad
                            hit_no_grad = True
                        else:
                            assert False, f"detect unexcepted context manager in first with code: {module_name}.{context_manager_name}"
            assert hit_no_grad, f"context manager torch.no_grad not existed"

        # with torch.no_grad() as _nnscaler_no_grad, torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True, cache_enabled=True) as _nnscaler_autocast:
        def second_with_node_check(node: ast.With):
            hit_no_grad = False
            hit_autocast = False
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    func = item.context_expr.func
                    if isinstance(func, ast.Attribute):
                        module_name = func.value.id if isinstance(func.value, ast.Name) else None
                        context_manager_name = func.attr
                        if module_name == 'torch' and context_manager_name == 'no_grad':
                            assert not hit_no_grad
                            hit_no_grad = True
                        elif module_name == 'torch' and context_manager_name == 'autocast':
                            assert not hit_autocast
                            hit_autocast = True
                        else:
                            assert False, f"detect unexcepted context manager in second with code: {module_name}.{context_manager_name}"
            assert hit_no_grad, f"context manager torch.no_grad not existed"
            assert hit_autocast, f"context manager torch.autocast not existed"

        with_node_count = 0
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.With):
                if with_node_count == 0:
                    first_with_node_check(node)
                elif with_node_count == 1:
                    second_with_node_check(node)
                else:
                    assert False, f"detect unexcepted third with code"
                with_node_count += 1


def get_gencode(cubesave_dir, module_class, index=0):
    from nnscaler.parallel import _PARALLEL_MODULE_NAMESPACE, _get_full_qualified_name, _DEFAULT_INSTANCE_NAME
    from pathlib import Path

    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}.{_DEFAULT_INSTANCE_NAME}'
    outdir: Path = cubesave_dir / Path(namespace.replace('.', '/').strip('/'))
    filecontent = (outdir /f'gencode{index}.py').read_text()
    return filecontent


def _train_cube_one_sample(model: ParallelModule, mbs):
    init_random()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    data = []
    init_random()
    data_size = mbs
    for _ in range(data_size):
        data.append(tuple(dummy_data().values()))
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        losses = model.train_step(x)
        print(f'loss {_}: {losses}')
        optimizer.step()
        gnorm = optimizer.clip_gnorm()
        grads = {n: p.grad for n, p in model.named_parameters()}
        model._add_extra_state(grads, '')
        weights = {n: p.data for n, p in model.named_parameters()}
        model._add_extra_state(weights, '')
        results.append(clone_to_cpu_recursively([grads, weights, gnorm]))
        optimizer.zero_grad()
    return results


def gpu_worker_cube_one_sample():
    init_distributed()
    init_random()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ctx_manager') as tempdir:
        init_random()
        model = CtxManagerModel()
        model = parallelize(
            model,
            dummy_data(),
            pas_policy='tp',
            compute_config= ComputeConfig(
                2, 2,
                use_end2end=True,
            ),
            gen_savedir=tempdir
        )
        model.cuda()
        train_result = _train_cube_one_sample(model, 1)
        return train_result


def _train_ga(model, update_freq, data_size):
    init_random()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = []
    init_random()
    for _ in range(data_size):
        data.append(dummy_data())
    results = []
    for i, x in enumerate(data):
        model.train()
        loss = model(**x)
        print(f'loss {i}: {loss}')
        loss.backward()
        if i % update_freq == update_freq - 1:
            optimizer.step()
            grads = {n: p.grad for n, p in model.named_parameters()}
            weights = {n: p.data for n, p in model.named_parameters()}
            # gnorm calculation doesn't support float64, so let's skip it
            results.append(clone_to_cpu_recursively([grads, weights, torch.tensor(0.0)]))
            optimizer.zero_grad()
    return results


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_loss_scaling():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    init_random()
    model = CtxManagerModel()
    ga4_result = _train_ga(model, 1, 1)
    assert len(ga4_result) == 1
    ga4_grads = ga4_result[0][0]

    cube2_results = launch_torchrun(2, gpu_worker_cube_one_sample)
    cube2_result = merge_cube_result({k: v for k, v in cube2_results.items()})
    assert len(cube2_result) == 1
    cube2_grads = cube2_result[0][0]
    assert len(cube2_grads) == len(ga4_grads)
    for k in cube2_grads:
        assert torch.allclose(cube2_grads[k].cpu(), ga4_grads[k].cpu(), atol=1e-6, rtol=1e-6)
