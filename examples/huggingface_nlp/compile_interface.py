#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from nnscaler.graph.parser.converter import to_fx_graph
import nnscaler
from nnscaler.runtime.utils import microbatches
from typing import Any, Dict
import inspect
import os
import logging


logger = logging.getLogger('compile_wrapper')


def prepare_dataloader(model, dummy_input):
    forward_signature = inspect.signature(model.forward)
    params_with_defaults = tuple(
        v.default if k not in dummy_input else dummy_input[k].to(torch.cuda.current_device())
        for k, v in forward_signature.parameters.items()
    )
    dataloader = microbatches([params_with_defaults] * 2)
    return dataloader


def calcu_max_diff(before_trace, after_trace):
    """Recursively calculate the max difference between two dicts or two tensors"""
    max_diff = 0
    if isinstance(after_trace, torch.Tensor):
        diff = torch.max(torch.abs(after_trace.to(torch.cuda.current_device()) - before_trace.to(torch.cuda.current_device())))
        if diff > max_diff:
            max_diff = diff
    elif isinstance(after_trace, dict):
        for key in after_trace.keys():
            diff = calcu_max_diff(before_trace[key], after_trace[key])
            if diff > max_diff:
                max_diff = diff
    elif isinstance(after_trace, (list, tuple)):
        for i in range(len(after_trace)):
            diff = calcu_max_diff(before_trace[i], after_trace[i])
            if diff > max_diff:
                max_diff = diff
    else:
        diff = calcu_max_diff(before_trace, after_trace)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TraceCompileException(Exception):
    """An exception that occurs during the model tracing or compilation process"""
    def __init__(self, message, original_exception):
        super().__init__(f"{message}: {str(original_exception)}")
        self.original_exception = original_exception


class ModelCompiler:
    def __init__(self, model: torch.nn.Module, dummy_input: Dict[str, Any], policy):
        nnscaler.init()
        self.model = model.to(torch.cuda.current_device())
        forward_signature = inspect.signature(model.forward)
        self.dummy_input = {
            k: v.default if k not in dummy_input
            else dummy_input[k].to(torch.cuda.current_device())
            for k, v in forward_signature.parameters.items()
        }
        self.policy = policy
        self.model.eval()
        self.before_trace = self.model(**self.dummy_input)

    def forward_diff(self, model):
        """Compute the model's output and compare it with the original model's output"""
        if model is None:
            raise RuntimeError("Model is None")
        model.to(torch.cuda.current_device())
        model.eval()
        _value = model(**self.dummy_input)
        max_diff = calcu_max_diff(self.before_trace, _value)
        return max_diff

    def trace(self):
        """Trace model"""
        try:
            if torch.cuda.is_available():
                try:
                    traced_gm = to_fx_graph(self.model, self.dummy_input)
                except:
                    raise
                logger.info("Successfully traced with gpu")
                return traced_gm
            else:
                raise RuntimeError("CUDA is not available")
        except Exception as e:
            raise TraceCompileException("An error occurred during trace the model.", e)

    def parallel(self, model):
        """Compile model"""
        from nnscaler.parallel import parallelize, ComputeConfig
        try:
            parallel_model = parallelize(
                model,
                self.dummy_input,
                pas_policy=self.policy,
                compute_config=ComputeConfig(1, 1),
                reuse='override',
                load_module=True,
            )
            return parallel_model
        except Exception as e:
            raise RuntimeError("An error occurred during the model compilation.", e)

    def train(self, model, steps = 1):
        """Train model with dummy_input for steps"""
        from torch.optim import SGD
        set_seed(0)
        model.to(torch.cuda.current_device())
        model.train()
        optimizer = SGD(model.parameters(), 1e-3)
        loss_fct = torch.nn.CrossEntropyLoss()
        label = torch.zeros_like(self.dummy_input['input_ids'])
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(**self.dummy_input)
            if isinstance(output, torch.Tensor):
                loss = loss_fct(output, label)
            elif isinstance(output, dict):
                if 'logits' in output:
                    loss = loss_fct(output['logits'].view(-1, output['logits'].shape[-1]), label.view(-1))
                elif 'last_hidden_state' in output:
                    loss = loss_fct(output['last_hidden_state'].view(-1, output['last_hidden_state'].shape[-1]), label.view(-1))
                else:
                    raise RuntimeError(f"Output keys doesn't supported: {output.keys()}")
            else:
                raise RuntimeError(f"Output type doesn't supported: {type(output)}")
            loss.backward()
            optimizer.step()
        return loss

    def export(self):
        """Trace the model using torch.export, similar to trace"""
        from torch.export import export
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['TORCH_LOGS'] = '+dynamo'
        os.environ['TORCHDYNAMO_VERBOSE'] = '1'
        try:
            if torch.cuda.is_available():
                try:
                    dummy_inputs = tuple(self.dummy_input.values())
                    exported_gm = export(self.model, self.dummy_input)
                except:
                    raise
                logger.info("Successfully export with gpu")
                return exported_gm
            else:
                raise RuntimeError("CUDA is not available")
        except Exception as e:
            raise TraceCompileException("An error occurred during export and forward the model.", e)
