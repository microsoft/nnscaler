# Tessel

Tessel is a schedule composer that searches efficient pipeline schedules for various operator placement strategies of large DNN models.

A large DNN model can be represented by a dataflow graph with operators as nodes and tensors as edges. Each operators can be placed on one arbitrary device or multiple devices (using tensor parallelism). Placing operators on devices generates an operator placement strategy, where input tensors are executed through these operators across multiple devices.

A training or inference iteration (i.e., mini-batch) may involve hundreds or even thousands of micro-batches. Micro-batches share a same operator placement strategy and are data-independent from each other. Therefore, there are numerous scheduling choices to decide execution ordering for operators from different micro-batches. Such execution ordering significantly impacts the final performance. To solve this problem, this project is proposed to search for high-efficient execution ordering (schedules) for various operator placement strategies.

## Install

```bash
pip install -e .
```

For runtime part, this repo is built on MagicCube of branch `zhiqi/main`.

## Examples

User can write various operator placement following the examples in `examples/simulator/cases_tessel.py`.

To generate schedules that are feasible for a schedule, use `Composer.compose`.

User can firstly try out with an example like:

```bash
python examples/simulator/cases_tessel.py \
    --placement mshape --ndevs 4 \
    --nmicros 6 --memory 6 \
    --save mshape.4stages.sched.json
```

## Tutorial

Generating a schedule involves two steps: 1) Specify an operator placememnt strategy; 2) Call `Composer` to search for efficient schedules.

### Step 1: Specify an Operator Placement

Tessel uses `Block` to represent a sub-graph of model. The model dataflow graph can be composed by several blocks. Each block can be associated with an execution time (integer), memory (positive or negative integer). Following examples determine a 1F1B-schedule placememnt (V-Shape).

```python
from tessel.schedule.schedplan import SchedPlan, Block

def vshape(ndevs: int) -> SchedPlan:
    """
    f             b
      f         b  
        f     b    
          f b      
    """
    sched = SchedPlan(ndevs)
    fblocks = [Block(0, span=1, memory=1, btype="Forward") for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks = [Block(0, span=2, memory=-1, btype="Backward") for _ in range(ndevs)]
    bdevs = [[devid] for devid in range(ndevs)][::-1]
    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched

placement = vshape(ndevs=4)  # 4-device v-shape placement
```

The `sched.add_block_seq` will add blocks into a schedule plan (currently the micro-batch plan). `blocks` will be connected with data dependency from the prior one to the next one. To specify more flexible data dependencies among blocks, please refer to interface of `Block.make_dependency`.

### Step 2: Search for Schedules

Then, search for a schedule plan for the vshape placemement:

```python
# maximal peak memory capacity for each device
memory = 4 
# problem size: how many micro-batches involve in the repetend construction.
nmicros = 4
# search
schedule = Composer.compose_n(micro, memory, nmicros)
print(schedule)
```


## Contributions

All contributions are welcomed! Please issue PR for new cases or new features. 

## License

This project is under MIT license.