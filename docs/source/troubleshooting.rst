###############
Troubleshooting
###############

Reuse Cache
===========

I have modified the model but the result does not change
--------------------------------------------------------

Remove ``.nnscaler`` directory in the working path and try again.

nnScaler's workflow is first compiling the model, and then running the compiled (generated) model.
After modifying the original model, you need to tell nnScaler to re-compile it.

This can be achieved by two ways:

1. Remove the compiled model (located in ``.nnscaler`` directory);
2. Set ``TrainerArgs.gen_reuse`` to ``"override"``.

We recommend to set ``gen_reuse="override"`` to debug the model,
and change it to ``gen_reuse="auto"`` for deployment.

.. code-block:: python

    trainer_args = TrainerArgs(
        gen_reuse='override',
        ...
    )
    trainer = Trainer(trainer_args=trainer_args)
    trainer.run()

Note that setting ``gen_reuse="match"`` will NOT solve this problem,
since it only checks ``compute_config``, not the model.

"RuntimeError: Output directory ... is not empty. And the existing files do not match..." after modifying models
----------------------------------------------------------------------------------------------------------------

As the error message said, please remove the ``.nnscaler`` directory.

To prevent this kind of errors permanently, you can set ``gen_reuse`` to ``"override"``, at the expense of time.

Example stacktrace: ::

    Traceback (most recent call last):
      File "train.py", line 244, in <module>
        main()
      File "train.py", line 240, in main
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 95, in run
        self._setup()
      File ".../nnscaler/cli/trainer.py", line 206, in _setup
        pmodel_class = nnscaler.parallelize(
      File ".../nnscaler/parallel.py", line 983, in parallelize
        outdir, reusable = _prepare_and_check_reusable(gen_savedir, module_class, compute_config, instance_name, reuse)
      File ".../nnscaler/parallel.py", line 547, in _prepare_and_check_reusable
        raise RuntimeError(f'Output directory {outdir} is not empty. '
    RuntimeError: Output directory .../.nnscaler/_parallel_modules/__main__/WrapperModel/_ is not empty. And the existing files do not match with current config. You can remove the directory and try again, or set reuse to ReuseType.NONE/ReuseType.OVERRIDE to regenerate the code.

Known Issues
============

"KeyError: '__mro__'" and errors mentioning "_dynamo"
-----------------------------------------------------

Add ``import torch._dynamo`` to the beginning of your main script.

Due to a limitation in nnScaler, the dynamic import of ``torch._dynamo`` cannot be correctly traced.
This can be workaround by importing it before tracing.

Example stacktrace: ::

    Traceback (most recent call last):
      File "train.py", line 286, in <module>
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 95, in run
        self._setup()
      File ".../nnscaler/cli/trainer.py", line 206, in _setup
        pmodel_class = nnscaler.parallelize(
      File ".../nnscaler/parallel.py", line 993, in parallelize
        regen_status = _gencode(
    
    ......

      File ".../site-packages/transformers/models/llama/modeling_llama.py", line 1041, in _update_causal_mask
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
      File ".../nnscaler/graph/parser/fx/concrete_trace_utils/operator_patcher.py", line 354, in patch_run
        return new_func(*args, **kwargs)
      File ".../site-packages/transformers/modeling_attn_mask_utils.py", line 259, in _ignore_causal_mask_sdpa
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
      File ".../nnscaler/graph/parser/fx/concrete_trace_utils/operator_patcher.py", line 354, in patch_run
        return new_func(*args, **kwargs)
      File ".../site-packages/torch/__init__.py", line 2003, in __getattr__
        return importlib.import_module(f".{name}", __name__)
      File ".../importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)

    ......

      File ".../site-packages/torch/_dynamo/utils.py", line 567, in unwrap_with_attr_name_if_wrapper
        elif is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
      File ".../inspect.py", line 1738, in getattr_static
        if not _is_type(obj):
      File ".../inspect.py", line 1707, in _is_type
        _static_getmro(obj)
      File ".../inspect.py", line 1685, in _static_getmro
        return type.__dict__['__mro__'].__get__(klass)
    KeyError: '__mro__'

"ModuleNotFoundError: No module named 'nnscaler.autodist.dp_solver'" when using editable install
------------------------------------------------------------------------------------------------

Run the following command: ::

    python -c 'import os,sys,nnscaler,cppimport.import_hook ; sys.path.append(os.path.dirname(nnscaler.__path__[0])) ; import nnscaler.autodist.dp_solver'

If it complains ``GLIBCXX_x.y.z`` not found, check the next issue.

Example stacktrace: ::

    Traceback (most recent call last):
      File "model.py", line 48, in <module>
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 95, in run
        self._setup()
      File ".../nnscaler/cli/trainer.py", line 206, in _setup
        pmodel_class = nnscaler.parallelize(
      File ".../nnscaler/parallel.py", line 988, in parallelize
        regen_status = _gencode(
      File ".../nnscaler/parallel.py", line 753, in _gencode
        graph = pas_policy(graph, compute_config)
      File ".../nnscaler/policies.py", line 303, in pas_autodist
        return parallelize_graph(graph, autodist_cfg)
      File ".../nnscaler/autodist/apis.py", line 117, in parallelize_graph
        search_out = calc_parallel_plan(graph, autodist_config)
      File ".../nnscaler/autodist/apis.py", line 98, in calc_parallel_plan
        pp_out = calc_optimal_spmd_plan(autodist_graph, autodist_config)
      File ".../nnscaler/autodist/spmd_solver.py", line 1503, in calc_optimal_spmd_plan
        spmd_outs = spmd_solver.solve([(0, model_graph.op_num - 1)], 1)[0]
      File ".../nnscaler/autodist/spmd_solver.py", line 1374, in solve
        return self.do_dp(intervals, topk)
      File ".../nnscaler/autodist/spmd_solver.py", line 1183, in do_dp
        import nnscaler.autodist.dp_solver as dp_solver
    ModuleNotFoundError: No module named 'nnscaler.autodist.dp_solver'

"ImportError: ...... libstdc++.so.6: version \`GLIBCXX_x.y.z' not found"
-------------------------------------------------------------------------

This is caused by gcc and glibc version mismatch.
Typically it means it's using the system gcc and conda's glibc.

You can remove conda's glibc to force it use system glibc: ::

    rm <PATH_TO_CONDA_ENV>/lib/libstdc++.so.6

The path is shown in the error message.

Example stacktrace: ::

    $ python -c 'import nnscaler,cppimport.import_hook ; import nnscaler.autodist.dp_solver'
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: /home/user/miniconda3/envs/user/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by .../nnscaler/autodist/dp_solver.cpython-310-x86_64-linux-gnu.so)

Incorrect Usages
================

"RuntineError: Loss can only be scalar tensor ..." when forward returns dict
----------------------------------------------------------------------------

When using nnScaler's Trainer, the return value of the top-level ``forward()`` must not be a dict.
It can either be:

1. A loss tensor;
2. A tuple where the first element is a loss tensor.

Detailed explaination: :ref:`end2end model <end2end>`.

How to fix:

.. code-block:: diff

    def forward(self, data):
        ...
        -return {'loss': loss, 'ntokens': ntokens}
        +return loss, ntokens

Example stacktrace: ::

    Traceback (most recent call last):
      File "example.py", line 27, in <module>
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 95, in run
        self._setup()
      File ".../nnscaler/cli/trainer.py", line 206, in _setup
        pmodel_class = nnscaler.parallelize(
      File ".../nnscaler/parallel.py", line 988, in parallelize
        regen_status = _gencode(
      File ".../nnscaler/parallel.py", line 737, in _gencode
        graph, forward_args = _gen_graph(
      File ".../nnscaler/parallel.py", line 656, in _gen_graph
        raise RuntimeError(f"Loss can only be scalar tensor but got {ir_loss.shape if isinstance(ir_loss, IRTensor) else ir_loss}")
    RuntimeError: Loss can only be scalar tensor but got {'loss': t1596(p920,(1,),d(),v(0/1)), 'ntokens': t1597(p922,(1,),d(),v(0/1))}

"TypeError: ... 'device_type' must be str, not ConcreteAttrProxy" when using torch>=2.4
---------------------------------------------------------------------------------------

nnScaler does not support torch 2.4 yet.
Downgrade to torch 2.3.* will fix the issue: ::

    pip install "torch<2.4"

Example stacktrace: ::

    Traceback (most recent call last):
      File "model.py", line 43, in <module>
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 95, in run
        self._setup()
      File ".../nnscaler/cli/trainer.py", line 206, in _setup
        pmodel_class = nnscaler.parallelize(
      File ".../nnscaler/parallel.py", line 988, in parallelize
        regen_status = _gencode(

    ......

      File ".../nnscaler/graph/parser/fx/concrete_trace_utils/operator_patcher.py", line 354, in patch_run
        return new_func(*args, **kwargs)
      File ".../torch/amp/autocast_mode.py", line 237, in __init__
        if not is_autocast_available(self.device):
      File ".../torch/amp/autocast_mode.py", line 36, in is_autocast_available
        return torch._C._is_autocast_available(device_type)
    TypeError: _is_autocast_available(): argument 'device_type' (position 1) must be str, not ConcreteAttrProxy

Flash Attention Problems
========================

"NameError: name 'flash_attn' is not defined"
---------------------------------------------

When using flash attention, it must be registered with ``register_op`` API.
Check :doc:`the llama 3 example <examples/llama>` for its usage.

Example stacktrace: ::

    Traceback (most recent call last):
      File "train.py", line 247, in <module>
        trainer.run()
      File ".../nnscaler/cli/trainer.py", line 98, in run
        self._train()
      File ".../nnscaler/cli/trainer.py", line 558, in _train
        self._train_epoch(epoch)
      File ".../nnscaler/cli/trainer.py", line 698, in _train_epoch
        losses = self.model.train_step(batches, is_dummy_batch)
      File ".../nnscaler/runtime/module.py", line 967, in train_step
        output = self._train_step(dataloader)
      File ".nnscaler/_parallel_modules/__main__/WrapperModel/_/gencode0.py", line 1228, in _train_step
        cross_entropy_1433, getitem_62_1431 = nnscaler.runtime.executor.fexecute('segment1977', model.segment1977, *(data_1780, ), requires_grad=True)
      File ".../nnscaler/runtime/executor.py", line 105, in fexecute
        outputs = subgraph(*input_dtensors)
      File ".nnscaler/_parallel_modules/__main__/WrapperModel/_/gencode0.py", line 452, in segment1977
        add_7_2220, add_7_2221 = ckpt.checkpoint(recompute, unsqueeze_1439, embedding_2130, embedding_2131, use_reentrant=False)
      File ".../site-packages/torch/_compile.py", line 24, in inner
        return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
      File ".../site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
        return fn(*args, **kwargs)
      File ".../site-packages/torch/_dynamo/external_utils.py", line 36, in inner
        return fn(*args, **kwargs)
      File ".../site-packages/torch/utils/checkpoint.py", line 494, in checkpoint
        ret = function(*args, **kwargs)
      File ".nnscaler/_parallel_modules/__main__/WrapperModel/_/gencode0.py", line 386, in recompute
        apply_1495 = flash_attn.flash_attn_interface.FlashAttnFunc.apply(transpose_4_1492, transpose_5_1493, transpose_6_1494, ifexpr_930, None, True, (-1, -1), 0.0, None, False, False)
    NameError: name 'flash_attn' is not defined

"ImportError" when using flash attention
----------------------------------------

This is likely an error in flash attention itself.
Please try the related import command outside nnScaler.
If it still fails, please refer to `flash attention <https://github.com/Dao-AILab/flash-attention>`_'s docs.

If your ``flash-attn`` package is installed from pip,
you can try to use a wheel its `release page <https://github.com/Dao-AILab/flash-attention/releases>`_
which matches your environment more accurately.

Example stacktrace: ::

    Traceback (most recent call last):
      File "train.py", line 9, in <module>
        from modeling_modifier import nnscaler_llama_init
      File "modeling_modifier.py", line 14, in <module>
        from transformers.models.llama.modeling_llama import LlamaAttention, LLAMA_ATTENTION_CLASSES, apply_rotary_pos_emb, LlamaRMSNorm
      File ".../site-packages/transformers/models/llama/modeling_llama.py", line 53, in <module>
        from flash_attn import flash_attn_func, flash_attn_varlen_func
      File ".../site-packages/flash_attn/__init__.py", line 3, in <module>
        from flash_attn.flash_attn_interface import (
      File ".../site-packages/flash_attn/flash_attn_interface.py", line 10, in <module>
        import flash_attn_2_cuda as flash_attn_cuda
    ImportError: .../site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv

Hugging Face Access
===================

"Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted. ... Please log in."
---------------------------------------------------------------------------------------

You need to request for `Llama 3 access <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_ on Hugging Face first.
Once you get access, generates your `Hugging Face token <https://huggingface.co/docs/hub/security-tokens>`_ and export it: ::

    export HF_TOKEN=hf_...

.. (FIXME: check it) Or alternatively, you can try replacing ``meta-llama/Meta-Llama-3-8B-Instruct`` with ``microsoft/Phi-3-mini-4k-instruct``.
