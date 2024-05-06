

## Profling

Window attention relies on a relative postion bias, which is indexed from
the input with concrete value. Since profiler cannot construct correct value scope, please remove the relative operators during profiling.
