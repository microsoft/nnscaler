#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List

from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import IRAdapterPrim

from nnscaler.codegen.syntax.blocks import ClassBlock, FunctionBlock

from nnscaler.codegen.emit import FuncEmission


class AutogradAdapterCodeGen(FuncEmission):
    """
    Generate autograd adapter code (PyTorch)
    """
    def __init__(self):

        self.fw_ins: List[IRSubTensor] = list()
        self.fw_body: List[str] = list()
        self.fw_ous: List[IRSubTensor] = list()

        self.bw_ins: List[IRSubTensor] = list()
        self.bw_body: List[str] = list()
        self.bw_ous: List[IRSubTensor] = list()

    def emit_prim(self, prim: IRAdapterPrim) -> str:
        if len(prim.inputs()) == 1:
            itensors = self.tensor_name(prim.inputs()[0])
        else:
            itensors = self.tuple_name(prim.inputs())
        kwargs = list()
        for name, val in prim.kwargs.items():
            kwargs.append(f'{name}={val}')
        kwargs = ', '.join(kwargs)
        outputs = self.return_name(prim.outputs())
        code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
        return code

    def gen(self, fadapter: IRAdapter) -> List[str]:
        assert fadapter.isfw() and fadapter.differentiable and fadapter.custom, "generate autograd for a non-differentiable adapter"
        assert fadapter.mirror is not None
        name = self.name(fadapter)
        with ClassBlock(class_name=name, derived=['torch.autograd.Function']) as cb:
            # forward
            cb.insert_body('@staticmethod')
            finputs = [self.tensor_name(t) for t in fadapter.inputs()]
            with FunctionBlock(func_name='forward', args=['ctx']+finputs) as fw:
                for prim in fadapter.prims:
                    fw.insert_body(self.emit_prim(prim))
                outputs = self.return_name(fadapter.outputs())
                fw.insert_body(f'return {outputs}')
            cb.insert_body(fw.code)
            # backward
            cb.insert_body('@staticmethod')
            badapter: IRAdapter = fadapter.mirror
            binputs = [self.tensor_name(t) for t in badapter.inputs()]
            with FunctionBlock(func_name='backward', args=['ctx']+binputs) as bw:
                for prim in badapter.prims:
                    bw.insert_body(self.emit_prim(prim))
                outputs = self.return_name(badapter.outputs())
                bw.insert_body(f'return {outputs}')
            cb.insert_body(bw.code)
        return cb.code

    def name(self, adapter: IRAdapter) -> str:
        return f'Adapter{adapter.cid}'
