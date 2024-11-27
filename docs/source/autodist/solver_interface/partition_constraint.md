# Control partitions of a model in AutoDist

In autodist, we provide a set of partition constraints to control the distributed plan of the model. The partition constraints are specified in a yaml file. The following is an example of the partition constraints.

```yaml
- allowed_partition_dims:
  - 0,0
  name: torchscale.component.xmoe.routing.top2gating
  parent_module: Top2Gate
  replica_allowed: false
- allowed_partition_dims:
  - 0,0
  name: torchscale.component.xmoe.moe_layer.dispatch_expert_inputs
  parent_module: MOELayer
  replica_allowed: false
- allowed_partition_dims:
  - 0,0
  name: torchscale.component.xmoe.moe_layer.merge_expert_outputs
  parent_module: MOELayer
  replica_allowed: false
- allowed_partition_dims:
  - 0,0
  name: torchscale.component.xmoe.routing.compute_logits
  parent_module: Top2Gate
  replica_allowed: true
```

In this example, we have four partition constraints for the MoE model in retnet. Each partition constraint has 4 fields: `name`, `parent_module`, `allowed_partition_dims`, and `replica_allowed`.

- `name` is the name of the corresponding operator in the model. It equals to the `signature` field in the `IRFwOperation` in nnScaler. Note: signature is the full name of the operator, for example, you should provide `torch.nn.functional.linear` instead of `linear`.
- `parent_module` is the **closest** father module name of the operator. You can provide two partition constraints with a same `name` but different `module` to control the partition of the same operator in different modules. Similar to `recompute_modules`, Module name can be any suffix of the full module name, e.g., `module1` will match `x.module1`, `y.module1`, `x.module1` will match `x.module1` but not `y.module1`.
- `allowed_partition_dims` is a list of allowed partition dimensions of input tensors. Each element in the list is a list of two integers, which are the index of the partitioned tensor among inputs and the partitioned dimension of that tensor. For example, the annotation of `torchscale.component.xmoe.routing.compute_logits` can be `(C 16) E^ C, E^ C M^ -> (C 16) M^`. `allowed_partition_dims = [[0, 0]]` means we only allow to partition the first input tensor along the first dimension, which is `(C, 16)` in this case. An empty list means no partition is allowed, note that in yaml, you should give an empty list explicitly, i.e., `allowed_partition_dims: []`.
- `replica_allowed` is a boolean value. If it is `true`, it is allowed to replicate the operator across devices.

After specifying the partition constraints in a yaml file, we can feed them to autodist by `--autodist-partition-constraints-path <absolute-path-to-yaml-file>` in fairseq.

# Examples

Three examples are provided in `pc_examples` folder.

- `docs/source/autodist/solver_interface/pc_examples/retnet_dp_pc.yaml` helps to generate a pure data parallel plan.
- `docs/source/autodist/solver_interface/pc_examples/retnet_mp_pc.yaml` helps to generate a pure model parallel plan.
- `docs/source/autodist/solver_interface/pc_examples/retnet_hybrid_pc.yaml` helps to generate a hybrid plan: data parallel for the attention module and model parallel for the feed forward module.
