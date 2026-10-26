#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from nnscaler.graph.parser.frame import Frame
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.module import CubeModule


class AttrModule(CubeModule):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer('offset', torch.zeros(2))
        self.add_full_map('weight', 1, True, 'weight', (4, ), (slice(1, 3), ),
                          1)
        self.add_full_map('offset', 2, False, 'offset', (2, ), (slice(0, 2), ),
                          1)


def _create_sentinel(path: str):
    Path(path).touch()


class UnsafePayload:

    def __init__(self, sentinel: Path):
        self.sentinel = sentinel

    def __reduce__(self):
        return _create_sentinel, (str(self.sentinel), )


class CustomTensor(torch.Tensor):
    pass


def test_load_attr_content_loads_tensor_shards(tmp_path):
    filename = str(tmp_path / 'fullmodel.pt')
    torch.save({1: torch.arange(4, dtype=torch.float32)}, filename + '.0')
    torch.save({2: torch.tensor([4.0, 5.0])}, filename + '.1')

    module = AttrModule()
    module.load_attr_content(filename)

    torch.testing.assert_close(module.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(module.offset, torch.tensor([4.0, 5.0]))


def test_load_attr_content_rejects_unsafe_pickle(tmp_path):
    filename = str(tmp_path / 'fullmodel.pt')
    sentinel = tmp_path / 'deserialized'
    torch.save({1: UnsafePayload(sentinel)}, filename + '.0')

    with pytest.raises(RuntimeError, match='Cannot safely load model parameter chunk'):
        CubeModule().load_attr_content(filename)

    assert not sentinel.exists()


def test_load_attr_content_warns_for_unsafe_torch_version(tmp_path, monkeypatch):
    filename = str(tmp_path / 'fullmodel.pt')
    torch.save(
        {
            1: torch.arange(4, dtype=torch.float32),
            2: torch.tensor([4.0, 5.0]),
        },
        filename + '.0',
    )
    monkeypatch.setattr(torch, '__version__', '2.5.1')

    with pytest.warns(RuntimeWarning, match='known remote code execution vulnerability'):
        module = AttrModule()
        module.load_attr_content(filename)

    torch.testing.assert_close(module.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(module.offset, torch.tensor([4.0, 5.0]))


def test_save_attr_content_normalizes_tensor_subclasses(tmp_path):
    filename = tmp_path / 'fullmodel.pt'
    tensor = IRFullTensor((4, ), name='weight')
    value = torch.arange(4, dtype=torch.float32).as_subclass(CustomTensor)
    frame = Frame()
    frame.add_attr(tensor, value, 'weight')

    frame.save_attr_content(filename)

    saved_value = torch.load(str(filename) + '.0', weights_only=True)[tensor.tid]
    assert type(saved_value) is torch.Tensor
    torch.testing.assert_close(
        saved_value,
        torch.arange(4, dtype=torch.float32, device='cpu'),
    )


def test_save_attr_content_preserves_sparse_tensors(tmp_path):
    filename = tmp_path / 'fullmodel.pt'
    tensor = IRFullTensor((4, ), name='weight')
    value = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]),
        torch.tensor([1.0, 3.0]),
        (4, ),
    )
    frame = Frame()
    frame.add_attr(tensor, value, 'weight')

    frame.save_attr_content(filename)

    saved_value = torch.load(str(filename) + '.0', weights_only=True)[tensor.tid]
    assert saved_value.layout is torch.sparse_coo
    torch.testing.assert_close(saved_value.to_dense(), value.to_dense())


def test_load_attr_content_rejects_legacy_tensor_subclass_shards(tmp_path):
    filename = str(tmp_path / 'fullmodel.pt')
    torch.save(
        {1: torch.arange(4, dtype=torch.float32).as_subclass(CustomTensor)},
        filename + '.0',
    )

    with pytest.raises(RuntimeError, match='Regenerate it with this nnScaler version'):
        AttrModule().load_attr_content(filename)
