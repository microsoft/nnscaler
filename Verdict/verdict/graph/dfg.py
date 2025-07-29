from typing import Dict, List, TYPE_CHECKING
from abc import ABC, abstractmethod


from .elements import Node, Tensor
from .world import World, DTag

if TYPE_CHECKING:
    from verdict.operators.names import OpName


class DFG(ABC):
    @property
    @abstractmethod
    def ID(self) -> str: ...

    @property
    @abstractmethod
    def W(self) -> World: ...

    @abstractmethod
    def nodes(self) -> List[Node]: ...

    @abstractmethod
    def node_dtag(self) -> DTag: ...

    @abstractmethod
    def node_kwargs(self, node: Node) -> Dict: ...

    @abstractmethod
    def node_inputs(self, node: Node) -> List[Tensor]: ...

    @abstractmethod
    def node_outputs(self, node: Node) -> List[Tensor]: ...

    @abstractmethod
    def node_opname(self, node: Node) -> "OpName": ...

    @abstractmethod
    def tensors(self) -> List[Tensor]: ...

    @abstractmethod
    def tensor_shape(self, tensor: Tensor) -> List[int]: ...

    @abstractmethod
    def is_initialized(self, tensor: Tensor) -> bool: ...
