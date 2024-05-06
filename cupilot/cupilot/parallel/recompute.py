# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Callable, Tuple, Any

import torch


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue
            x = inp.detach()
            x.requires_grad = True # inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


class ChainRecompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fns: List[Callable], *args):

        ctx.run_functions = fns
        ctx.preserve_rng_state = True

        # preserve rng state
        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.fwd_gpu_state = torch.cuda.get_rng_state()

        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = args
            for fn in fns:
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                outputs = fn(*outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument."
            )
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        devid = torch.cuda.current_device()
        with torch.random.fork_rng(devices=(devid,)):
            # TODO: check rng state correctness under 
            # multiple forward and backward context
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                torch.cuda.set_rng_state(ctx.fwd_gpu_state)

            num_fns = len(ctx.run_functions)
            output_grads = grads
            for nruns in range(num_fns - 1, -1, -1):
                # infer to get nruns-th function inputs
                fn_inputs = tuple(inputs)
                with torch.no_grad():
                    for fn in ctx.run_functions[:nruns]:
                        # print(f'backward: infer {fn.__name__}')
                        fn_inputs = fn(*fn_inputs)
                        if isinstance(fn_inputs, torch.Tensor):
                            fn_inputs = (fn_inputs,)
                # forward with autograd enabled
                fn_inputs = detach_variable(fn_inputs)
                # print(f'backward: forward {ctx.run_functions[nruns].__name__}')
                with torch.enable_grad():
                    fn_outputs = ctx.run_functions[nruns](*fn_inputs)
                # backward
                # print(f'backward: backward {ctx.run_functions[nruns].__name__}')
                if isinstance(fn_outputs, torch.Tensor):
                    fn_outputs = (fn_outputs,)
                for t in fn_outputs:
                    assert t.requires_grad, fn.__name__
                torch.autograd.backward(fn_outputs, output_grads)
                output_grads = tuple(
                    inp.grad if isinstance(inp, torch.Tensor) else None
                    for inp in fn_inputs
                )
        return (None,) + output_grads


def chain_recompute(functions: List[Callable], *args):
    """Chain-based recompute

    The output of the prior function is used as the exact input of 
    the next function.
    
    Note:
        We assume every output tensor of each function requires gradient.
    """
    return ChainRecompute.apply(functions, *args)
