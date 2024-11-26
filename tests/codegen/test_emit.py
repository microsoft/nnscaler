#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from nnscaler.codegen.emit import CodeEmission, IRValue
from nnscaler.ir.cten import IRObject
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function import Dropout
from nnscaler.ir.tensor import IRFullTensor


def test_tensor_name():
    repr_expr = CodeEmission().tensor_name
    assert repr_expr(1, 'model.') == '1'
    assert repr_expr('1') == "'1'"

    assert repr_expr(IRObject('name', 111, 'value'), 'model.') == 'name_111'
    assert repr_expr(IRObject('name', 111, 'value').as_attr(), 'model.') == 'model.name_111'
    assert repr_expr((IRObject('name', 111, 'value').as_attr(),), 'model.') == '(model.name_111,)'

    assert repr_expr(slice(1, None, IRObject('name', 111, 'value').as_attr()), 'model.') == 'slice(1, None, model.name_111)'
    assert repr_expr({'a': 1, 'b': IRObject('name', 111, 'value')}, 'model.') == "{'a': 1, 'b': name_111}"
    assert repr_expr([1], 'model.') == '[1]'
    assert repr_expr((1,), 'model.') == '(1,)'

    assert repr_expr((1,...), ) == '(1, Ellipsis)'

    with pytest.raises(ValueError):
        from datetime import datetime
        repr_expr(datetime.now())



def test_emit_module_attr():
    dropout = Dropout(IRFullTensor([1024, 1024], requires_grad=True), p=0.5, training=IRValue('self.training'), signature='torch.nn.functional.dropout')
    code = FuncEmission().emit_fnode(dropout, runtime_devid=0, plan_ndevs=1, runtime_ndevs=1)
    print(code)
    assert 'training=self.training' in code[0]
