##########
Quickstart
##########

************
Installation
************

nnScaler can be installed from GitHub:

.. code-block:: bash

    pip install https://github.com/microsoft/nnscaler/releases/download/0.6/nnscaler-0.6-py3-none-any.whl

    # You may also want to clone the repo to try out the examples
    git clone --recursive https://github.com/microsoft/nnscaler

***************************
Parallelize a Minimal Model
***************************

You can verify the installation by parallize a minimal model:

.. code-block:: python

    # model.py

    import os
    import torch
    from nnscaler.cli.trainer import Trainer
    from nnscaler.cli.trainer_args import *
    from nnscaler.utils import set_default_logger_level
    
    set_default_logger_level('INFO')
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 10)
    
        def forward(self, data):
            x = self.linear(data['x'])
            return torch.nn.functional.cross_entropy(x, data['y'])
    
    class RandomDataset:
        def __init__(self, split):
            pass
    
        def __getitem__(self, i):
            return {
                'x': torch.rand(100),
                'y': torch.randint(10, tuple()),
            }
    
        def __len__(self):
            return 100
    
    if __name__ == '__main__':
        world_size = int(os.getenv('WORLD_SIZE', 1))
        trainer_args = TrainerArgs(
            compute_config=ComputeConfig(plan_ngpus=1, runtime_ngpus=world_size, use_end2end=True),
            model=ModelConfig(type=Model),
            optimizer=OptimizerConfig(type=torch.optim.AdamW),
            dataset=DatasetConfig(type=RandomDataset, train_args={'split': 'train'}),
            max_train_steps=10,
            enable_progress_bar=False,
        )
        trainer = Trainer(train_args=trainer_args)
        trainer.run()

To run it in parallel, use `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_: ::

    torchrun --nproc_per_node=2 model.py

Expected output:

.. (FIXME: adjust log level)

::

    2024-09-09 20:28:04 | INFO | nnscaler.graph.parser.converter | constant folding disabled to parse graph
    2024-09-09 20:28:04 | WARNING | nnscaler.graph.graph | nnScaler does not support to compute gradients for IRPyFunc.
    Following nodes require gradients, this may trigger error in backward:
            _operator.getitem, cid: 1
    
    2024-09-09 20:28:04 | WARNING | nnscaler.graph.segment | nnScaler does not support backward of IRPyFunc: PyOp1-()(sign=getitem, inputs=(Object(data35, val={'x': t32(p30,(1, 100),d(),v(0/1)), 'y': t34(p33,(1,),d(),v(0/1))}, is_constant=False), 'x'), outputs=(t25(p4,(1, 100),d(),v(0/1)),)), skip setting gradient, please register it as IRDimOps.
    2024-09-09 20:28:04 | INFO | nnscaler.autodist.apis | AutoDistConfig {'pc_path': '', 'profile_dir': PosixPath('/home/.cache/nnscaler/autodist/1.0/NVIDIA_RTX_A6000'), 'topk': 20, 'task_name': '__1gpus_1update_freq', 'load_plan_path': None, 'save_plan_path': None, 'consider_mem': True, 'zero_stage': 0, 'zero_ngroups': 1, 'opt_resident_coef': 2, 'opt_transient_coef': 0, 'is_train': True, 'mesh_desc': MeshDesc(row=1, col=1), 'ngpus': 1, 'recompute_modules': '', 'memory_constraint': 40802189312, 'memory_granularity': 524288, 'micro_batch_size': 1, 'update_freq': 1, 'world_size': 1, 'nproc': 1, 'ignore_small_tensor_threshold': 524288, 'verbose': False, 're_profile': False, 'pipeline': False, 'pipeline_pivots': '', 'pipeline_nstages': 1, 'pipeline_scheduler': '1f1b', 'max_pipeline_bubble_ratio': 0.2, 'max_pipeline_unbalance_ratio': 0.5, 'solver': 'dp', 'parallel_profile': True, 'transient_mem_coef': 2}
    2024-09-09 20:28:04 | WARNING | nnscaler.autodist.cost_database | Communication profile data not found, using default data at /home/nnscaler/nnscaler/resources/profile/mi200/comm
    2024-09-09 20:28:04 | INFO | nnscaler.autodist.cost_database | Profiling in parallel
    2024-09-09 20:28:06 | INFO | nnscaler.autodist.cost_database | device 0 finished profiling 1 nodes
    2024-09-09 20:28:06 | INFO | nnscaler.autodist.cost_database | device 2 finished profiling 0 nodes
    2024-09-09 20:28:06 | INFO | nnscaler.autodist.cost_database | device 1 finished profiling 1 nodes
    2024-09-09 20:28:06 | INFO | nnscaler.autodist.cost_database | device 3 finished profiling 0 nodes
    2024-09-09 20:28:07 | WARNING | nnscaler.autodist.model_graph | detect a non-IRDimops _operator.getitem at File "/home/nnscaler/test.py", line 16, in forward,  x = self.linear(data['x']) that produces tensors
    2024-09-09 20:28:07 | WARNING | nnscaler.autodist.model_graph | detect a non-IRDimops _operator.getitem at File "/home/nnscaler/test.py", line 17, in forward,  return torch.nn.functional.cross_entropy(x, data['y']) that produces tensors
    2024-09-09 20:28:07 | INFO | nnscaler.autodist.model_graph |
    -------------------------nnScaler Graph Profiling Result-------------------------
    
    depth 1
        param_mem - [('linear, Linear', '0.00 MB'), ('_operator.getitem', '0.00 MB'), ('_operator.getitem', '0.00 MB')]
        fw_span - [('torch.nn.functional.cross_entropy', '0.08 ms'), ('linear, Linear', '0.08 ms'), ('_operator.getitem', '0.00 ms')]
        train_mem - [('linear, Linear', '0.00 MB'), ('torch.nn.functional.cross_entropy', '0.00 MB'), ('_operator.getitem', '0.00 MB')]
        buffer_mem - [('_operator.getitem', '0.00 MB'), ('linear, Linear', '0.00 MB'), ('_operator.getitem', '0.00 MB')]
    depth 2
        param_mem - [('torch.nn.functional.linear', '0.00 MB')]
        fw_span - [('torch.nn.functional.linear', '0.08 ms')]
        train_mem - [('torch.nn.functional.linear', '0.00 MB')]
        buffer_mem - [('torch.nn.functional.linear', '0.00 MB')]
    
    2024-09-09 20:28:07 | INFO | nnscaler.autodist.apis | param mem 0 MB, buff mem 0 MB, activation mem 0 MB
    2024-09-09 20:28:07 | INFO | nnscaler.autodist.apis | estimated minimum memory per device 0.0 MB
    2024-09-09 20:28:07 | INFO | nnscaler.autodist.spmd_solver | no partition constraint is loaded
    2024-09-09 20:28:07 | INFO | nnscaler.autodist.cost_database | Profiling in parallel
    2024-09-09 20:28:08 | INFO | nnscaler.autodist.cost_database | device 1 finished profiling 1 nodes
    2024-09-09 20:28:08 | INFO | nnscaler.autodist.cost_database | device 3 finished profiling 0 nodes
    2024-09-09 20:28:08 | INFO | nnscaler.autodist.cost_database | device 2 finished profiling 0 nodes
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.cost_database | device 0 finished profiling 1 nodes
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | force_replica_threshold is 0
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | finish building op partitions
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | finish building following relationships
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | finish filtering useless partitions
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | total state num is 4
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | output each operator's importance ratio (percentages of states that can be reduced by forcing the operator to be partitioned in a single partition)
    
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.spmd_solver | finish spmd solver initializetion
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.apis | use plan with e2e time/s 0.94ms
    2024-09-09 20:28:09 | INFO | nnscaler.autodist.apis |
    autodist plan analysis for stage 0 on devices [0] with mem 0.00 GB:
    
    Total computation time: 0.94 ms
    Top 10 of operators that consume the most computation time:
        torch.nn.functional.cross_entropy: 0.50 ms
        torch.nn.functional.linear: 0.44 ms
        _operator.getitem: 0.00 ms
    Top 10 of operators computation time sum: 0.94 ms
    
    Top 2 operators split info:
        torch.nn.functional.cross_entropy:
            FwOp4-()(name=cross_entropy, inputs=(t28(p10,(1, 10),d(),v(0/1)), t29(p12,(1,),d(),v(0/1))), outputs=(t24(p13,(1,),d(),v(0/1)),))
            File "/home/nnscaler/test.py", line 17, in forward,  return torch.nn.functional.cross_entropy(x, data['y'])
            N^ C^, N^ -> 1^, OpPartition((-1,), (1,)), comp_time: 0.50 ms, comm_time: 0.00 ms
    
    
        torch.nn.functional.linear:
            FwOp2-()(name=linear, inputs=(t25(p4,(1, 100),d(),v(0/1)), w26(p6,(10, 100),d(),v(0/1)), w27(p8,(10,),d(),v(0/1))), outputs=(t28(p10,(1, 10),d(),v(0/1)),))
            File "/home/nnscaler/test.py", line 16, in forward,  x = self.linear(data['x'])
            a k^, n k^, n -> a n, OpPartition((-1,), (1,)), comp_time: 0.44 ms, comm_time: 0.00 ms
    
    
    Total communication time: 0.00 ms
    Top 10 operators that consume the most communication time:
    Top 10 of operators communication time sum: 0.00 ms
    
    Module analysis:
    Depth 1:
        Top 3 modules that consume the most computation time:
        Top 3 modules that consume the most communication time:
        Top 3 modules that consume the most memory:
    Depth 2:
        Top 3 modules that consume the most computation time:
        Top 3 modules that consume the most communication time:
        Top 3 modules that consume the most memory:
    
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish reordering producer and consumer
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish removing anchor nodes
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish replacing auto pyfunc
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish transforming multiref nodes
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish local fusion & multiref for 4 tensors
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish reordering producer and consumer
    2024-09-09 20:28:09 | INFO | nnscaler.graph.gener.gen | finish generating 4 activation adapters
    2024-09-09 20:28:09 | INFO | nnscaler.execplan.planpass.fusion | adapter fusion: successfully fuse 0 differentiable adapters
    2024-09-09 20:28:09 | INFO | nnscaler.runtime.module | loading partitioned model from /home/nnscaler/.nnscaler/_parallel_modules/__main__/Model/_/fullmodel.pt, number of model parameter chunks: 1
    2024-09-09 20:28:09 | INFO | nnscaler.cli.trainer | Training...
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Epoch 0: 010/100 train_loss=2.261, lr=0.001, gnorm=5.590, train_wall=0.004
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Saving checkpoint after 10 steps with loss=2.261.
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Saving checkpoint to checkpoints/0000-0010
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Saving checkpoint as the last checkpoint.
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Best loss updated: inf -> 2.261
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Saving checkpoint as the best checkpoint.
    2024-09-09 20:28:10 | INFO | nnscaler.cli.trainer | Reached max train steps(10): Training is done.

*********
Next Step
*********

The above example uses nnScaler's :doc:`Trainer APIs <trainer>`.
To learn more about it, you may check our :doc:`Llama 3 example <examples/llama3_demo>`.

Or if you prefer to use a familiar trainer, we also provides integration with `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_.
The usage is demostrated by :doc:`nanoGPT example <examples/nanogpt>`.

If you want to try a more advanced model, please check :doc:`Llama 3 128K sequence length example <examples/llama>`.
