#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import os
from pathlib import Path

import pytest
import torch

from nnscaler.parallel import ComputeConfig, _prepare_namespace, parallelize, broadcast_weights

from .common import init_distributed
from ..launch_torchrun import launch_torchrun


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _to_cube_model(module, compute_config, cube_savedir,
    instance_name=None, load_module=False,
    broadcast_strategy='none',
    **kwargs,
):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        'tp',
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name,
        load_module=load_module,
        broadcast_strategy=broadcast_strategy,
        **kwargs,
    )


def _gpu_worker(tmp_path):
    init_distributed()
    world_size = torch.distributed.get_world_size()
    local_world_size = world_size // 2
    # fake two machines, as we use different cube_savedir for each worker
    os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
    tempdir = tmp_path / f'worker_{torch.distributed.get_rank() // local_world_size}'
    node_rank = torch.distributed.get_rank() // local_world_size

    # from nnscaler.runtime.device import DeviceGroup
    # # create groups
    # for i in range(local_world_size):
    #     group_ranks = list(range(i, world_size, local_world_size))
    #     DeviceGroup().get_group(group_ranks)

    p = lambda t, b, i, load_module=True, **kwargs: _to_cube_model(
            Module(),
            ComputeConfig(1, world_size),
            t,
            load_module=load_module,
            broadcast_strategy=b,
            instance_name=i,
            **kwargs,
        )
    # case 1: no broadcast, so only rank 0 can load the module
    #         rank 1 will raise ModuleNotFoundError
    # this will hang forever due to the distributed group creation in generated code.
    # if node_rank == 0:
    #     p(tempdir, 'none', '_1')
    # else:
    #     with pytest.raises(ModuleNotFoundError):
    #         p(tempdir, 'none', '_1')

    # case 2: broadcast only code, so only rank 0 can load the module
    #         rank 1 will raise FileNotFoundError because it will fail to load attr_map files and more
    if node_rank == 0:
        p(tempdir, 'code', '_2')
    else:
        with pytest.raises(FileNotFoundError):
            p(tempdir, 'code', '_2')

    # case 3: broadcast except weights, so only rank 0 can load the module
    #         rank 1 will raise RuntimeError because it will fail to load fullmodel.pt
    if node_rank == 0:
        p(tempdir, 'no_weights', '_3')
    else:
        with pytest.raises(RuntimeError, match='Cannot find file.*'):
            p(tempdir, 'no_weights', '_3')

    # case 4: broadcast except weights, every rank can succeed if don't lood init params
    m = p(tempdir, 'no_weights', '_4',
            init_module_params=torch.distributed.get_rank() == 0
    )
    if node_rank == 0:
        for n, pa in m.named_parameters():
            if n.startswith('linear_weight'):
                pa.data.fill_(1.0)
    else:
        for n, pa in m.named_parameters():
            if n.startswith('linear_weight'):
                assert not torch.equal(pa.data, torch.ones_like(pa.data))
    broadcast_weights(m)
    # check if broadcast_weights works
    for n, pa in m.named_parameters():
        if n.startswith('linear_weight'):
            assert torch.equal(pa.data, torch.ones_like(pa.data))

    # case 5: broadcast all, all ranks will succeed
    p(tempdir, 'all', '_5')

    # case 6: test incremental broadcast
    # generate without broadcasting
    _, outdir6 = _prepare_namespace(tempdir, Module, '_6')
    m = p(tempdir, 'none', '_6', load_module=False)
    if node_rank != 0:
        assert list(Path(outdir6).glob('*')) == []

    # case 6.1: broadcast code even we set broadcast_strategy to `all`
    # because only code is new generated.
    m = p(tempdir, 'all', '_6', load_module=False, reuse='graph')
    if node_rank != 0:
        # only python files are broadcasted
        assert set(f.name for f in Path(outdir6).glob('**/*') if f.is_file()) == set(
            [f'gencode{i}.py' for i in range(world_size)] + ['compute_config.pt']
        )

    torch.distributed.barrier()

    # case 6.2: everything should be broadcasted, including weights
    # so the load_module will succeed.
    m = p(tempdir, 'all', '_6', load_module=True, reuse='override')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_broadcast(tmp_path):
    launch_torchrun(2, _gpu_worker, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_broadcast4(tmp_path):
    launch_torchrun(4, _gpu_worker, tmp_path)
