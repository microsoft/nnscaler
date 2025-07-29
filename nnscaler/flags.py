"""
Environment flags for compiling options
"""

import os


def _to_bool(s: str) -> bool:
    val = os.environ.get(s, default=0)
    return bool(int(val))


def _to_int(s: str, default=0) -> int:
    val = os.environ.get(s, default=default)
    return int(val)


class CompileFlag:
    # ================ compiling ========================
    # worker sleep in seconds
    worker_sleep = _to_int('WORKER_SLEEP')
    disable_intra_rvd = _to_bool('DISABLE_INTRA_RVD')
    disable_inter_rvd =  _to_bool('DISABLE_INTER_RVD')
    disable_comm_fusion = _to_bool('DISABLE_COMM_FUSION')

    visualize_plan = _to_bool('VISUALIZE_PLAN')

    # ============ code generation ===============
    use_nnfusion = _to_bool('USE_NNFUSION')
    use_jit = _to_bool('USE_JIT')
    disable_code_line_info = _to_bool('DISABLE_CODE_LINE_INFO')  # will add original code information in generated code, note that this will make trace slow

    # ============== runtime ====================
    dev_mode = _to_bool('SINGLE_DEV_MODE')  # allow to use python xx.py
    async_comm = _to_bool('ASYNC_COMM')
    line_timer = _to_bool('LINE_TIMER')

    # ============== reducer ==================
    # use zero optimization on optimizer status.
    # to cooperate with zero, user needs to call `model.parameters_for_optimizer()`
    # to get parameters for optimizer, and `model.gather_params()` after `optimizer.step()`
    use_zero = _to_bool('USE_ZERO')
    # use async communication to overlap gradient synchronization and backward computation
    async_reducer = _to_bool('ASYNC_REDUCER')  # use async reducer
    # maximal reducer weight bytes for one allreduce (only effective for async): default 128MB
    max_reducer_bucket = _to_int('MAX_REDUCER_BUCKET', default=137217728)
    # perform reducer op on gradients, can be sum, avg, mean, max, min. Default is sum
    reducer_op = os.environ.get('REDUCER_OP', default='sum')
    # zero_ngroups is the number of subgroups in each original ZeRO gruop (e.g., weights reducer)
    # ZeRO subgroup is obtained by dividing the original ZeRO group by zero_ngroups
    # it helps reduce communication cost of allgather weights in ZeRO, but increase the weights'
    # optimization states on each GPU.
    zero_ngroups = _to_int('ZERO_NUM_GROUPS', default=1)

    # use automate mixture precision training, where weights, gradients
    # and optimizer status are kept in its original data type (can be float32),
    # but some of the forward operators will be converted to float16.
    use_amp = _to_bool('USE_AMP')


class RuntimeFlag:

    # if True, skip model.zero_grad().
    # when applying gradient accumulation,
    # this flag should be set to True at the first accumulation step,
    # and set to False at other accumulation steps.
    # By default False, which means the gradients of parameters in the reducers
    # will be zeroed at the beginning of every iteration.
    skip_zero_grad: bool = False

    # if True, skip reducer.sync_grads().
    # when applying gradient accumulation,
    # this flag should be set to True at the last accumulation step,
    # .and set to False at other accumulation steps.
    # By default False, which means the gradients will be reduced at the end of every iteration.
    skip_reducer: bool = False
