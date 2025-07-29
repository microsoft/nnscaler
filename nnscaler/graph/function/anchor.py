
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRObject


class IRGraphAnchor(IRFwOperation):
    """
    The anchor function serves for
        1) navigation inside the graph
        2) user hints of staging boundary inside the graph

    This operator will eventually be removed from graph,
    user doesn't need to manipulate it.

    To add anchor node in the graph, a user can simply insert anchor
    function in model forward like following:

    ```python
    class Model(torch.nn.Module):

        def __init__(self):
            xxx

        def forward(self, x):
            for layer in self.layers:
                nnscaler.runtime.function.anchor('layer start')
                x = layer(x)
            return x
    ```

    Then there will be anchor nodes named `layer start` inserted inside graph.
    Policy maker can quickly access them by

    ```python
    graph.select(name='layer start')`
    ```

    Or quickly find all anchor nodes through

    ```python
    anchors = graph.select(ntype=IRGraphAnchor)
    ```
    """
    def __init__(self, signature: str, name: str):
        super().__init__(name, signature, [], 1)
        self.kwargs['name'] = name
        self.set_output(0, IRObject('anchor', value=None))

    def infer_shape(self):
        return True

    def __repr__(self) -> str:
        return f"AnchorOp-{self.cid}(name={self.name})"
