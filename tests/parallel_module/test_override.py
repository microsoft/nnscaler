#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
from time import sleep
import sys
import tempfile
import pytest
import torch
import shutil

from nnscaler.graph.parser import FxModuleParser
from nnscaler.parallel import ReuseType, parallelize, ComputeConfig, _load_parallel_module_class
from nnscaler.runtime.module import ParallelModule

from ..utils import new_empty, replace_all_device_with


def _to_cube_model(model_class, compute_config, cube_savedir, reuse, instance_name, load_module=True):
    parallelize(
        model_class,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'data',
        compute_config,
        reuse=reuse,
        gen_savedir=cube_savedir,
        instance_name=instance_name,
        load_module=False,
    )
    if load_module:
        module_class = _load_parallel_module_class(
            model_class,
            gen_savedir=cube_savedir,
            instance_name=instance_name,
            rank=0
        )
        m = new_empty(module_class, device='cpu', init_params=True)
        return m


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


@replace_all_device_with('cpu')
def test_override():
    with tempfile.TemporaryDirectory() as tempdir:
        # MATCH   | empty | generate
        cmodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'mm0')
        # MATCH  | match | do nothing
        cmodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'mm0')
        for (n1, v1), (n2, v2) in zip(cmodule1.named_parameters(), cmodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        # MATCH  | match | do nothing
        cmodule3 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'test')
        cmodule4 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, 'match', 'test')

        for (n1, v1), (n2, v2) in zip(cmodule3.named_parameters(), cmodule4.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule3_p = dict(cmodule3.named_parameters())
        keys = cmodule3_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule3_p[key]) for key in keys)

        # MATCH  | unmatch | raise error
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'm0')
        with pytest.raises(RuntimeError, match='.*not empty.*'):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, 'match', 'm0')

        # MOO   | empty | generate
        omodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o0')
        # MOO  | match | do nothing
        omodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o0')
        for (n1, v1), (n2, v2) in zip(omodule1.named_parameters(), omodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        # MOO  | unmatch | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o1', load_module=False)
        _to_cube_model(MyModule, ComputeConfig(2, 2, constant_folding=True),tempdir, ReuseType.MOO, 'o1')

        # MOO  | imported | raise error
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o2', load_module=True)
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, ReuseType.MOO, 'o2')

        # OVERRIDE   | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'mm0')

        # OVERRIDE   | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'test')

        # OVERRIDE  | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, ReuseType.OVERRIDE, 'test')

        # OVERRIDE   | empty | generate
        cmodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'test2')
        module_path = Path(sys.modules[cmodule1.__module__].__file__).parent
        test3_module_path = module_path.with_name('test3')
        test3_module_path.mkdir(exist_ok=True, parents=True)
        test4_module_path = module_path.with_name('test4')
        test4_module_path.mkdir(exist_ok=True, parents=True)
        test5_module_path = module_path.with_name('test5')
        test5_module_path.mkdir(exist_ok=True, parents=True)
        for f in module_path.glob('*'):
            if f.is_file():
                shutil.copy(f, test3_module_path / f.name)
                shutil.copy(f, test4_module_path / f.name)
                shutil.copy(f, test5_module_path / f.name)
        # fake two gpus
        shutil.copy(test4_module_path / 'gencode0.py', test4_module_path / 'gencode1.py')
        shutil.copy(test5_module_path / 'gencode0.py', test5_module_path / 'gencode1.py')

        # OVERRIDE   | match | generate
        cmodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, ReuseType.OVERRIDE, 'test3')
        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule1_p = dict(cmodule1.named_parameters())
        keys = cmodule2_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule1_p[key]) for key in keys)

        # OVERRIDE   | unmatch | generate
        assert (test4_module_path / 'gencode1.py').exists()
        cmodule3 = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'override', 'test4')
        assert not (test4_module_path / 'gencode1.py').exists()

        # Graph | matched | generate
        assert (test5_module_path / 'gencode1.py').exists()
        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        args_stat = (test5_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert not (test5_module_path / 'gencode1.py').exists()
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns

        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        (test5_module_path / 'forward_args.pkl').unlink()  # remove foward_args.pkl will force to generate new code
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').exists()

        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        attrmap_stat = (test5_module_path / FxModuleParser.ATTR_MAP_FILE).stat()
        (test5_module_path / FxModuleParser.ATTR_CONTENT_FILE_0).unlink()  # remove fullmodel.pt.0 will force to generate new code
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (test5_module_path / FxModuleParser.ATTR_MAP_FILE).stat().st_mtime_ns != attrmap_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').exists()

        # Graph | empty | generate
        g6_module = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g6')

        # Graph | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g6')

        # Graph | unmatch | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g7', False)
        g7_module_path = module_path.with_name('g7')
        graph_stat = (g7_module_path / 'graph.ckp').stat()
        args_stat = (g7_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2, constant_folding=True), tempdir, 'graph', 'g7', False)
        assert (g7_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (g7_module_path / 'forward_args.pkl').stat().st_mtime_ns != args_stat.st_mtime_ns

        # Graph | graph match | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g8', False)
        g8_module_path = module_path.with_name('g8')
        assert ComputeConfig.safe_load_from_file(g8_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(1, 1)
        graph_stat = (g8_module_path / 'graph.ckp').stat()
        args_stat = (g8_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2), tempdir, 'graph', 'g8', False)
        assert (g8_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (g8_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns
        assert ComputeConfig.safe_load_from_file(g8_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(2, 2)

        # MOO | graph match | generate code only
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'moo', 'g9', False)
        g9_module_path = module_path.with_name('g9')
        assert ComputeConfig.safe_load_from_file(g9_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(1, 1)
        graph_stat = (g9_module_path / 'graph.ckp').stat()
        args_stat = (g9_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2), tempdir, 'moo', 'g9', False)
        assert (g9_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (g9_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns
        assert ComputeConfig.safe_load_from_file(g9_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(2, 2)
