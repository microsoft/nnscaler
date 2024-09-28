#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

class IDGenerator:
    """
    Tensor / Operator manager. To guarantee that each IRTensor / IROperator id
    is unique and progressively increases.
    
    This class is designed in singleton pattern.
    """
    class __IDGenerator:
        def __init__(self):

            self._tensor_id = 0
            self._cell_id = 0

    instance = None

    def __init__(self):
        if not IDGenerator.instance:
            IDGenerator.instance = IDGenerator.__IDGenerator()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def gen_tensor_id(self):
        self.instance._tensor_id += 1
        return self.instance._tensor_id

    def gen_cell_id(self):
        self.instance._cell_id += 1
        return self.instance._cell_id

    def get_states(self):
        return (self._tensor_id, self._cell_id)
    
    def load_states(self, states: tuple):
        IDGenerator.instance._tensor_id = states[0]
        IDGenerator.instance._cell_id = states[1]

    def clear(self):
        self.instance._tensor_id = 0
        self.instance._cell_id = 0
