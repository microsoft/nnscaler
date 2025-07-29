from typing import Any, List
import logging
from nnscaler.utils import print_each_rank
import torch

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def memory_summary():
    torch.cuda.synchronize()
    # memory measurement
    mem = torch.cuda.max_memory_allocated()
    # mem = torch.cuda.max_memory_reserved()
    print_each_rank(
        '{:.2f} GB memory consumption'.format(mem / 1024 / 1024 / 1024),
        logger=_logger
    )
    
    return mem


def model_summary(model: torch.nn.Module, inputs: List[Any], do_eval=False, max_depth=6):
    """
    Benchmakr memory consumption for each module.
    This could only be called before any other forward/backward

    New attributes will be assigned to each module:

    * _summary_depth (Int)
    * _summary_begin_end (Boolean)
    * _summary_memory_state (Int)

    Make sure all of these attributes are not used in modules.
    """
    torch.cuda.empty_cache()
    static_memory = torch.cuda.memory_allocated()
    print_each_rank(
        'static model: {:,.2f} MB'.format(static_memory / 1024 / 1024),
        rank_only=0, logger=_logger)
    nparams = sum([param.numel() for param in model.parameters()])
    print_each_rank(
        'model paramters: {:,.2f} M'.format(nparams / 1000000),
        rank_only=0, logger=_logger)

    stat = dict(depth=0)
    def before_forward(module, input):
        module._summary_depth = stat['depth']
        module._summary_begin_end = False
        if len(list(module.children())) != 0:
            if stat['depth'] + 1 < max_depth:
                name = module.__class__.__name__
                module._summary_begin_end = True
                prefix = '   ' * module._summary_depth + '[Begin] > '
                print_each_rank(prefix + '{}:'.format(name), rank_only=0, 
                                logger=_logger)
        if module._summary_depth < max_depth:
            module._summary_memory_state = torch.cuda.memory_allocated()
        stat['depth'] += 1


    def after_forward(module, input, output):
        stat['depth'] -= 1
        if module._summary_depth >= max_depth:
            return
        name = module.__class__.__name__
        torch.cuda.empty_cache()
        curr_memory = torch.cuda.memory_allocated()
        mem_consumption = curr_memory - module._summary_memory_state
        mem_consumption = mem_consumption / 1024 / 1024

        n_params = sum([p.data.numel() for p in list(module.parameters())])
        
        prefix = '   ' * module._summary_depth
        prefix += '[End] > ' if module._summary_begin_end else '> '
        print_each_rank(
            prefix + '{}: Mem {:,.2f} MB, Params: {:,} ({:,.2f} MB if fp32)'.format(
                name, mem_consumption, n_params, n_params / 1024 / 1024 * 4),
            rank_only=0, logger=_logger)

    handle_pre = torch.nn.modules.module.register_module_forward_pre_hook(before_forward)
    handle_after = torch.nn.modules.module.register_module_forward_hook(after_forward)

    if do_eval:
        model.eval()
    else:
        model.train()
    _ = model(*inputs)

    handle_pre.remove()
    handle_after.remove()

    if stat['depth'] != 0:
        raise ValueError("Internal Error: depth {} not to 0".format(stat['depth']))
