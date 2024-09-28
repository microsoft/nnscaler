#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.execplan import ExecutionPlan


class PlanPass:

    @staticmethod
    def apply(execplan: ExecutionPlan) -> ExecutionPlan:
        raise NotImplementedError
