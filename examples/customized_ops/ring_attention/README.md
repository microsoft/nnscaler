# ring attention

Tensor parallel (partition head) is a widely used distributed plan to train large language models. Computation and memory are 
distributed evenly across devices. However, when the sequence length is extremely long (e.g., 1M), the partition degree of 
tensor parallel is constrained by the number of kv heads, which means that the maximum number of devices in a data parallel 
unit is no more than the number of kv heads. As a result, tensor parallel fails to scale a model with long sequence length.

[ring attention](https://arxiv.org/abs/2310.01889) is proposed to address this issue. It partitions q, k and v along the 
sequence dimension and passes the partitioned q, k and v through a ring of devices. [ring flash attention](https://github.com/zhuzilin/ring-flash-attention)
implements a high-performance version in PyTorch. This example attempts to integrate the causal version of ring attention 
(zigzag ring attention) into nnScaler.

The interface is wrapped in `zigzag_attn.py`. [flash attention](https://github.com/Dao-AILab/flash-attention) is required for this example.

In addition to the zigzag version, we also include a implementation based on [llama 3.1](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)'s technical report. This version uses `all_gather` and `reduce_scatter` to collect and distribute the kv values and gradients. You can check the code in `ring_attn.py`.

Test can be run with the following command:
```bash
torchrun --nproc_per_node 4 test_ring_attn.py
torchrun --nproc_per_node 4 test_zigzag_attn.py
```