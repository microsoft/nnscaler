#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Dimension Annotion Operations.

An operator has (multiple) input tensors and (multiple) output tensors.
Each tensor can be annotated with dimension annotations (DimAnno) using `identifiers`.
The same `identifier` indicates the they have the same real length.

* Dimension Annotation:

  e.g., 'a+', 'ab^', 'cd', '(ab+ c^ d)', '64'

A dimension of a tensor can be annotated by {identifier}{reduction} template.

An `identifier` must be one of:
  1) symbolic annotation that must match with the criteria of python str.isidentifier.
  2) numeric string that must match with python str.isdecimal. This indicates the shape is the same value
     numeric string will always have '^' reduction type'

Special identifier:
  1) '*': this special identifier indicates the dimension is dynamic, which will automatically get expanded given the shape
  2) '?': this special identifier indicates the value is can only be replicated, no matter it is a tensor or a non-tensor.

A `reduction` can be a set of {'', '+', '^'}:
  '' indicates this dimension can be partitioned, and each output should have this dimension.
  '+' indicates this dimension can be partitioned, and each output doesn't have this and need to do sum-reduction.
  '^' means this dimension cannot be partitioned.

A dimension can also be annotated with inner-dimensions using brackets, i.e., '(' and ')'.
The value of inner dimension needs to be inferrable, or indicated by function args (of same name).

* Shape Annotation:

  e.g., 'a (c+ d^) e'

A shape annotation consists of dimension annotation separated by (multiple) spaces.


* Operator Annotation:

  e.g., 'm k+, n k+ -> m n', '4 k+, k+ d -> 8 d', '* d^, s -> * s'

  An operator can be annotated with input shape annotations and output shape annotations.

  '->' seperates the inputs (left) and outputs (right) and ',' separates each input and output tensor.

  Identifiers in output tensor annotation needs to be
  1) apearred in input tensor annotations
  2) using numeric string

* Operator Partitioning Rule:

  1) Spatial Partition (dimension with '' reduce type):
      tensors can be uniformly partitioned on dimensions having spatial reduction type.
      other tensors in the operator that don't have this dimension will be replicated.

  2) Value Partition (dimension with '+' reduce type):
      * tensors can be uniformly partition on dimensions having numerical reduction type
      * other tensors in the the operator that don't have this dimension will be partitioned numerically.

  3) Illegal Splitting (dimension with '^' reduce type):
      * tensors can not be partitioned on dimensions having '^' reduction type.

"""

from typing import Callable, Dict, Iterable, List, Union, Set, Tuple, Optional
import enum
import re
import string
import logging
from itertools import dropwhile

from nnscaler.ir.cten import IRTensor, IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.algorithm.factory import DistAlgorithmFactory


_kSpecialIdentifiers = ('*', '?')
_logger = logging.getLogger(__name__)


class DimAnno:
    """
    To represent a dimension, name = {identifier}{reducetype}
    e.g.,
        ab^ means the dimension name is 'ab' and is a frozen dimension (cannot be split)
        ab+ means the dimension name is 'ab' and this dimension is a reduce dimension
        ['b', 'c+', 'd^'] means the dimension is composed by b, c, d
        where b can be spatially partitioned (apear in output), c is a reduce dimension,
        d is a frozen dimension (cannot be split)
    """

    class ReduceType(enum.Enum):
        Dim = ''
        Sum = '+'
        Freeze = '^'  # the dim is not allowed to be split

    def __init__(self, name: Union[str, Tuple[str]]):
        identifiers, reduces = DimAnno.parse(name)
        self._identifiers: Tuple[str] = identifiers
        self._reduces: Tuple[DimAnno.ReduceType] = reduces

    @property
    def name(self) -> str:
        """
        Return identifier without reduce
        """
        if len(self._identifiers) == 1:
            return self._identifiers[0]
        return '(' + ' '.join(self._identifiers) + ')'

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return self._identifiers

    @property
    def reduces(self) -> Tuple[ReduceType, ...]:
        return self._reduces

    def __eq__(self, other):
        if isinstance(other, DimAnno):
            if other.name == self.name:
                return True
        return False

    def __repr__(self):
        name_reduce = [name + reduce.value for name, reduce in zip(self._identifiers, self._reduces)]
        if len(name_reduce) == 1:
            return name_reduce[0]
        return '(' + ' '.join(name_reduce) + ')'

    @staticmethod
    def parse(anno: Union[str, Tuple[str]]) -> Tuple[Tuple[str], Tuple[ReduceType]]:
        assert isinstance(anno, str) or all(isinstance(n, str) for n in anno), \
            "Expect anno to be str or Tuple[str]"
        if isinstance(anno, str):
            anno = (anno,)
        if len(anno) > 1 and any(i in anno for i in _kSpecialIdentifiers):
            raise ValueError(f"Dim annotation cannot have {_kSpecialIdentifiers} as partial dimension.")
        identifiers, reduces = [], []
        for identifier in anno:
            # get reduce type
            reduce = DimAnno.ReduceType.Dim
            if identifier[-1] == DimAnno.ReduceType.Sum.value:
                reduce = DimAnno.ReduceType.Sum
                identifier = identifier[:-1]
            elif identifier[-1] == DimAnno.ReduceType.Freeze.value:
                reduce = DimAnno.ReduceType.Freeze
                identifier = identifier[:-1]
            # get identifier name
            assert str.isdecimal(identifier) or str.isidentifier(identifier) or identifier in _kSpecialIdentifiers, \
                f"identifier can only be integer or python identifier but got {identifier}"
            # integer will always have stay reduction type
            if str.isdecimal(identifier) or identifier == '?':
                reduce = DimAnno.ReduceType.Freeze
            identifiers.append(identifier)
            reduces.append(reduce)
        return tuple(identifiers), tuple(reduces)


class ShapeAnno:
    """
    Shape annotation

    e.g., a (b+ dim)  d^
    """

    def __init__(self, dim_annos: Union[str, Tuple[DimAnno]]):
        assert isinstance(dim_annos, str) or all(isinstance(adim, DimAnno) for adim in dim_annos), \
            f"dim_annos must be str or Tuple[DimAnno] but got {dim_annos}"
        if isinstance(dim_annos, str):
            dim_annos = ShapeAnno.parse(dim_annos)
        self._dims: Tuple[DimAnno] = dim_annos

    @property
    def dims(self) -> Tuple[DimAnno, ...]:
        return self._dims

    @property
    def ndims(self) -> int:
        """!
        Get dimension number
        """
        return len(self._dims)

    def getdims(self, identifier: str) -> List[int]:
        """!
        Get dims that has the identifier

        @param identifier str: the query identifier

        @return dims List[int]: dimensions that contain the identifier
        """
        dims = []
        for dim, dim_anno in enumerate(self.dims):
            if identifier in dim_anno.identifiers:
                dims.append(dim)
        return dims

    def __getitem__(self, dim: int) -> DimAnno:
        assert isinstance(dim, int), "indexing only support int, but got {dim}"
        assert dim < len(self._dims), f"dim {dim} out of boudary {len(self._dims)}"
        return self._dims[dim]

    def __setitem__(self, index: int, dim_anno: Union[DimAnno, str]):
        assert isinstance(index, int), "Expected index to be int"
        assert isinstance(dim_anno, (DimAnno, str)), "Expected DimAnno or str"
        if isinstance(dim_anno, str):
            dim_anno = DimAnno(dim_anno)
        self._dims[index] = dim_anno

    def __repr__(self) -> str:
        return ' '.join(repr(dim) for dim in self._dims)

    @property
    def ignore(self) -> bool:
        """!
        Check if the shape should be ignored, i.e., annotation is '?'.

        @return is_ignored bool: True if the shape should ignore else False
        """
        return self.ndims == 1 and self._dims[0].name == '?'

    @staticmethod
    def parse(shape_anno: str) -> Tuple[DimAnno]:
        """
        Parse annotations like of a single shape, e.g.,
            a (b+ dim)  d^

        @param shape str: shape annotation

        @return dim_annos Tuple[DimAnno]: tuple of dimension annotations
        """
        # => ['a', '(', 'b+', 'dim', ')', 'd^']
        shapes = list()
        for group in re.split(r'\ +', shape_anno):
            if len(group) == 0:
                continue
            if '(' in group or ')' in group:
                for group in re.split(r'([\(\)])', group):
                    if len(group) != 0:
                        shapes.append(group)
            else:
                shapes.append(group)
        edims: List[List[str]] = list()
        current_identifier = list()
        bracket_group = False
        for w in shapes:
            if w == '(':
                assert not bracket_group, "Syntax Error: brackets inside brackets not allowed"
                bracket_group = True
                if len(current_identifier) > 0:
                    edims.append(current_identifier)
                current_identifier = list()
            elif w == ')':
                assert bracket_group, "Syntax Error: backets are not balanced at ("
                bracket_group = False
                if len(current_identifier) > 0:
                    edims.append(current_identifier)
                current_identifier = list()
            else:
                if bracket_group:
                    current_identifier.append(w)
                else:
                    if len(current_identifier) > 0:
                        edims.append(current_identifier)
                    current_identifier = [w]
        assert not bracket_group, "Syntax Error: brackets are not balanced at )"
        if len(current_identifier) != 0:
            edims.append(current_identifier)
        dim_annos = tuple(DimAnno(edim) for edim in edims)
        return dim_annos

    @staticmethod
    def create_shape_str(shape: Tuple[int], reduction: str = '', iterator: Optional[Iterable] = None) -> List[str]:
        """
        Create dimension string annotation given the shape and identity iterator
        e.g., ['a+', 'b+', 'c+']

        @param shape List[int]: tensor shape
        @param reduction (str): reduction type must be in '', '+' or '^'
        @param iterator Optional[Iterable]: identity iterators. If None, use string.ascii_lowercase

        @return strs List[str]: each element in strs represents a dimension
        """
        if iterator is None:
            iterator = iter(string.ascii_lowercase)
        return [next(iterator) + reduction for _ in range(len(shape))]


class OpAnno:
    """
    Operator annotation.

    e.g., a (b c) d+, d+ k -> a (b c) k

    """

    def __init__(self, anno: Union[str, Tuple[Tuple[ShapeAnno], Tuple[ShapeAnno]]]):
        assert isinstance(anno, str) or \
            (len(anno) == 2 and all(isinstance(ashape, ShapeAnno) for ashape in list(anno[0]) + list(anno[1]))), \
            "Expected anno to be str or (inputs: [ShapeAnno], outputs: [ShapeAnno])"
        if isinstance(anno, str):
            anno = OpAnno.parse(anno)
        inputs, outputs = anno
        self._inputs: Tuple[ShapeAnno] = tuple(inputs)
        self._outputs: Tuple[ShapeAnno] = tuple(outputs)
        self._identifiers: Dict[str, int] = dict()   # identifier -> dimension length
        self._reduces: Dict[str, DimAnno.ReduceType] = dict()  # identifier -> reducer
        self.reset_identifiers()

    @property
    def identifiers(self) -> Tuple[str]:
        """!
        Get all identifier set

        @return identifiers Set[str]
        """
        return tuple(self._identifiers.keys())

    def input(self, index:int) -> ShapeAnno:
        assert index < len(self._inputs), "index out of boundary"
        return self._inputs[index]

    def inputs(self) -> Tuple[ShapeAnno, ...]:
        return self._inputs

    def set_input(self, index: int, shape_anno: Union[str, ShapeAnno]):
        """
        set the shape of index-th input tensors
        """
        assert isinstance(shape_anno, (str, ShapeAnno)), f"must be str or ShapeAnno but got {shape_anno}"
        assert index is None or index < len(self._inputs), "index out of boundary"
        inputs = list(self._inputs)
        inputs[index] = shape_anno if isinstance(shape_anno, ShapeAnno) else ShapeAnno(shape_anno)
        self._inputs = tuple(inputs)

    def output(self, index:int) -> ShapeAnno:
        assert index < len(self._outputs), "index out of boundary"
        return self._outputs[index]

    def outputs(self) -> Tuple[ShapeAnno, ...]:
        return self._outputs

    def set_output(self, index: int, shape_anno: Union[str, ShapeAnno]):
        """
        set the shape of index-th input tensors
        """
        assert isinstance(shape_anno, (str, ShapeAnno)), f"must be str or ShapeAnno but got {shape_anno}"
        assert index is None or index < len(self._outputs), "index out of boundary"
        outputs = list(self._outputs)
        outputs[index] = shape_anno if isinstance(shape_anno, ShapeAnno) else ShapeAnno(shape_anno)
        self._outputs = tuple(outputs)

    def reset_identifiers(self):
        """!
        Reset identifier set.

        @return None
        """
        self._identifiers = dict()
        shape_annos = list(self._inputs) + list(self._outputs)
        for ashape in shape_annos:
            for adim in ashape.dims:
                for identifier, reduce in zip(adim.identifiers, adim.reduces):
                    self._identifiers[identifier] = None
                    # TODO: check consistency
                    self._reduces[identifier] = reduce
        for identifier in self._identifiers.keys():
            if str.isdecimal(identifier):
                self._identifiers[identifier] = int(identifier)

    def setlen(self, identifier: str, length: int, override=False) -> bool:
        """!
        Set identifier length

        @param identifier str: identifier name
        @param length int: the real length of identifier
        @param override bool: if True will always set length, else will check if the existing length matches the new length

        @return success True if sucessfully set else False
        """
        assert identifier in self._identifiers, f"{identifier} not int identifier set {self._identifiers}"
        if not override:
            prelen = self._identifiers[identifier]
            if prelen is not None and prelen != length:
                return False
        self._identifiers[identifier] = length
        return True

    def getlen(self, identifier: str) -> Optional[int]:
        """!
        Get identifier length

        @param identifier str: identifier name

        @return length Optional[int]: the length of identifier
        """
        assert identifier in self._identifiers, f"{identifier} not exists {set(self._identifiers.keys())}"
        return self._identifiers[identifier]

    def get_reduce(self, identifier: str) -> DimAnno.ReduceType:
        """
        Get identifier reduce type

        @param identifier str: identifier name

        @return reduce DimAnno.ReduceType
        """
        assert identifier in self._reduces, f"{identifier} not exists {set(self._reduces.keys())}"
        return self._reduces[identifier]

    def __repr__(self) -> str:
        inputs = ', '.join(repr(input) for input in self.inputs())
        outputs = ', '.join(repr(output) for output in self.outputs())
        return inputs + ' -> ' + outputs

    @classmethod
    def parse(cls, anno: str) -> Tuple[Tuple[ShapeAnno, ...], Tuple[ShapeAnno, ...]]:
        """!
        Parse op annotation string to input shape annos and output shape annos.

        Args:
            anno (str): operator annotation
        Returns:
            tuple[tuple[ShapeAnno, ...], tuple[ShapeAnno, ...]]: input shape annos and output shape annos
        """
        # to inputs and outputs
        if '->' not in anno:
            raise ValueError(f"Syntax Error: Expected -> in operator anno: {anno}")
        inputs, outputs = anno.split('->')

        inputs = inputs.strip()
        inputs = [] if len(inputs) == 0 else inputs.split(',')
        outputs = outputs.strip()
        outputs = [] if len(outputs) == 0 else outputs.split(',')
        # to ShapeAnnos
        inputs: Tuple[ShapeAnno] = tuple(ShapeAnno(shape) for shape in inputs)
        outputs: Tuple[ShapeAnno] = tuple(ShapeAnno(shape) for shape in outputs)
        cls._verify_and_fix_inner_dim_anno(anno, inputs, outputs)

        return inputs, outputs

    @classmethod
    def _verify_and_fix_inner_dim_anno(
        cls,
        anno: str,
        inputs: Tuple[ShapeAnno, ...],
        outputs: Tuple[ShapeAnno, ...]
    ):
        """
        Verify to make sure reduce type of annotations are consistent.
        We also force reduce type of all inner dimension identifiers to be freeze,
        Because we can't partition inner dimensions in current implementation.
        """
        # used to track reduce type of each identifier
        id_reduce_map: dict[str, DimAnno.ReduceType] = dict()
        for shape in inputs + outputs:
            for edim in shape.dims:
                for idx, identifier in enumerate(edim.identifiers):
                    if id_reduce_map.setdefault(identifier, edim.reduces[idx]) != edim.reduces[idx]:
                        raise ValueError(f"Reduce type of identifier {identifier} is not consistent")

        non_first_inner_dim_ids = cls._get_non_leading_anno_ids(*inputs, *outputs)
        updated_ids = set()

        for shape in inputs + outputs:
            for edim in shape.dims:
                reduces = []
                for idx, identifier in enumerate(edim.identifiers):
                    if identifier in non_first_inner_dim_ids and edim.reduces[idx] != DimAnno.ReduceType.Freeze:
                        updated_ids.add(identifier)
                        reduces.append(DimAnno.ReduceType.Freeze)
                    else:
                        reduces.append(edim.reduces[idx])
                # HACK: modify protected member to fix reduce type inplace
                edim._reduces = tuple(reduces)

        if updated_ids:
            _logger.debug(f"Inner dimensions {updated_ids} in {anno} are forced to be frozen because they can't be partitioned")

    @classmethod
    def _get_non_leading_anno_ids(cls, *shape_annos: ShapeAnno) -> Set[str]:
        """
        collect all unpartitioned identifiers in inner dimensions, which most are not in the first position.
        See `transform_space` and `_verify_and_fix_inner_dim_anno` for more information.
        """
        nonleading_ids = set()
        for shape in shape_annos:
            for dim, dim_anno in enumerate(shape.dims):
                for identifier in list(dropwhile(lambda x: x == '1', dim_anno.identifiers))[1:]:
                    if not str.isdecimal(identifier):
                        nonleading_ids.add(identifier)
        return nonleading_ids

    @classmethod
    def create_op_str(
        cls,
        ins: Tuple[Tuple[Union[str, Tuple[str]]]],
        ous: Tuple[Tuple[Union[str, Tuple[str]]]]
    ) -> str:
        """
        Create operator annotation string
        e.g.,
            ins = [ ['a', 'b', 'c+'], ['c+', ['d', 'e']] ]
            ous = [ ['a', 'b', 'd', 'e'] ]
        =>
            'a b c+, c+ (d e) -> a b d e'

        Args:
            ins (Tuple[Tuple[Union[str, Tuple[str]]]): input identifier list
            ous (Tuple[Tuple[Union[str, Tuple[str]]]): output identifier list

        Returns:
            str: operator annotation
        """
        in_annos = list()
        ou_annos = list()
        for shape in ins:
            flatten = list()
            if isinstance(shape, str):
                in_annos.append(shape)
                continue
            for edim in shape:
                if isinstance(edim, str):
                    flatten.append(edim)
                # List
                elif len(edim) == 1:
                    flatten.append(edim[0])
                else:
                    flatten.append('(' + ' '.join(edim) + ')')
            in_annos.append(' '.join(flatten))
        for shape in ous:
            flatten = list()
            for edim in shape:
                if isinstance(edim, str):
                    flatten.append(edim)
                # List
                elif len(edim) == 1:
                    flatten.append(edim[0])
                else:
                    flatten.append('(' + ' '.join(edim) + ')')
            ou_annos.append(' '.join(flatten))
        return ', '.join(in_annos) + ' -> ' + ', '.join(ou_annos)

    def transform_space(self) -> List[Tuple[int, int]]:
        """
        Get transformation space of the operator, the transformation space
        represents all configurations that can be segmented

        @return List[Tuple[int, int]]: list of (idx, dim)
        """
        # only the first identifier in a dim anno is partitionable
        # eg. (a b c) x -> (b x)
        # b, c or x can't be partitioned, because they are not in the first position
        # a special case is when the leading identifiers are '1'
        # for example
        # (1 a b) -> b or (1 1 a b) -> b
        # in both cases, a can be partitioned, but b can't

        # collect all unpartitioned identifiers that are not in first position
        nonleading_ids = self._get_non_leading_anno_ids(*self.inputs(), *self.outputs())

        visited : Set[str] = set()  # to remove equivalent configurations
        configs = []
        shapes = self.inputs()
        for idx, shape in enumerate(shapes):
            if shape.ignore: continue
            for dim, edim in enumerate(shape.dims):
                # this for loop just checks the first element.
                for identifier, reduce in dropwhile(lambda x: x[0] == '1', zip(edim.identifiers, edim.reduces)):
                    if identifier in visited: continue
                    visited.add(identifier)
                    if reduce != DimAnno.ReduceType.Freeze and identifier not in nonleading_ids:
                        configs.append((idx, dim))
                    break

        return configs


class DimopSplit:
    """
    Partition status of a tensor
    """
    def __init__(self, dims: Optional[Union[int, List[int]]] = None, r = False, v = False) -> None:
        """Dimension split config

        Args:
            dims (Optional[Union[int, List[int]]], optional): [description]. Defaults to None.
        """
        if isinstance(dims, int):
            dims = (dims,)
        elif isinstance(dims, Iterable):
            dims = tuple(sorted(dims))
        self.dims: Optional[Tuple[int]] = dims
        self.rep: bool = r
        self.val: bool = v

    def isR(self) -> bool:
        return self.rep

    def isD(self) -> bool:
        return self.dims is not None

    def isV(self) -> bool:
        return self.val

    def __eq__(self, other):
        if not isinstance(other, DimopSplit):
            return False
        if other.isR() and self.isR():
            return True
        if other.isD() and self.isD() and other.dims == self.dims:
            return True
        if other.isV() and self.isV():
            return True
        return False

    def __hash__(self) -> int:
        if self.isV():
            return -1
        elif self.isR():
            return -2
        else:
            return self.dims

    def __repr__(self) -> str:
        if self.isD():
            return f'D({self.dims})'
        if self.isR():
            return f'R'
        if self.isV():
            return f'V'
        return 'Unknown-DimopSplit'

    @staticmethod
    def R():
        return DimopSplit(r=True)

    @staticmethod
    def V():
        return DimopSplit(v=True)

    @staticmethod
    def D(dims: Union[int, List[int]]):
        return DimopSplit(dims=dims)


class TransformRule:
    """
    Partition rule
    """
    def __init__(
        self,
        irules: Tuple[DimopSplit],
        orules: Tuple[DimopSplit],
        kwarg_modifier: Optional[Callable[[Dict, int, Union[int, str], int, int], Dict]] = None,
    ) -> None:
        self._inputs = tuple(irules)
        self._outputs = tuple(orules)
        modifier = kwarg_modifier if kwarg_modifier is not None else TransformRule.default_modifier
        self._modifier = (modifier,)

    def inputs(self) -> Tuple[DimopSplit]:
        return self._inputs

    def input(self, idx: int) -> DimopSplit:
        return self._inputs[idx]

    def outputs(self) -> Tuple[DimopSplit]:
        return self._outputs

    def output(self, idx: int) -> DimopSplit:
        return self._outputs[idx]

    def modifier(self) -> Optional[Callable]:
        return self._modifier[0]

    def __repr__(self) -> str:
        inputs = ', '.join(repr(split) for split in self._inputs)
        outputs = ', '.join(repr(split) for split in self._outputs)
        return f'{inputs} -> {outputs}'

    @staticmethod
    def default_modifier(kwargs: Dict, idx: int, dim: Union[int, str], num: int, subnode_idx: int) -> Dict:
        return kwargs


class IRDimops(IRFwOperation):
    """
    Einstein-inspired notation operations
    """
    def __init__(self, create_fn: Callable, name: str,
                 signature: str, annos: Tuple[str],
                 inputs: List[Union[IRTensor, IRObject]],
                 transform_rules: Optional[Tuple[TransformRule]] = None,
                 **kwargs):
        """!
        Create a IRDimops

        @param signature str: operator signature
        @param annos List[str]: annotation candidates
        @param inputs List[IRTensor]: input tensor list
        @param name str: the name of the operator
        @param transform_rules: the special rules to partition the operator. Default None.
        @param kwargs: the kwarg non-tensor parameters
        """
        assert all(isinstance(anno, str) for anno in annos), "Expect annos to be List[str]"
        self._annos_candidates: List[str] = tuple(annos)
        self._anno: OpAnno = None
        self._iannos: List[ShapeAnno] = None
        self._oannos: List[ShapeAnno] = None
        self._trans_rules: Tuple[TransformRule] = tuple(transform_rules) if transform_rules is not None else ()
        self._create_fn: Tuple[Callable] = (create_fn,)

        for anno in self._annos_candidates:
            anno = OpAnno(anno)
            # expand * and check shape dimension consistency
            if self.align(signature, inputs, anno, kwargs):
                self._iannos = anno.inputs()
                self._oannos = anno.outputs()
                self._anno = anno
                break
        else:
            raise RuntimeError(
                f"no matching anno for given annos."
                f"op: {signature}\n"
                f"inputs: {tuple(t.shape if isinstance(t, IRTensor) else t for t in inputs)}\n"
                f"annos: {annos}\n"
                f"kwargs: {kwargs}\n"
            )

        n_outputs = len(self._oannos)
        super().__init__(name, signature, inputs, n_outputs, **kwargs)

        # change tensor to IRObject for '?' annotation
        for idx, shape_anno in enumerate(self._oannos):
            if shape_anno.ignore:
                self.set_output(idx, IRObject())

    @property
    def anno(self) -> OpAnno:
        return self._anno

    @property
    def transform_rules(self) -> Tuple[TransformRule]:
        return self._trans_rules

    def ianno(self, index: int) -> ShapeAnno:
        """!
        Get index-th input tensor shape annotation

        @param index int: the input index

        @return dim_annos ShapeAnno: a tuple that each element is a dimension annotation
        """
        assert index < len(self.inputs()), "index out of boudary"
        return tuple(self._iannos[index])

    def oanno(self, index: int) -> ShapeAnno:
        """!
        Get index-th output tensor shape annotation

        @param index int: the output index

        @return dim_annos ShapeAnno: a tuple that each element is a dimension annotation
        """
        assert index < len(self.outputs()), "index out of boudary"
        return self._oannos[index]

    def infer_shape(self) -> bool:
        """
        Shape inference using the matched annotation and tensor.

        @return sucess: True if successfully inferred shape
        """
        for oidx, otensor in enumerate(self.outputs()):
            shape_anno = self.oanno(oidx)
            if shape_anno.ignore:
                # otensor can be any type, including IRObject, collection types (list, dict, etc.)
                continue
            shape = []
            for odim in range(shape_anno.ndims):
                accum = 1
                for identifier in shape_anno[odim].identifiers:
                    accum *= self.anno.getlen(identifier)
                shape.append(accum)
            otensor.shape = shape
        # print(f'=> sign: {self.signature} anno: {self.anno}\n'
        #       f'=> inputs: {self.inputs()}\n'
        #       f'=> outputs: {self.outputs()}')
        return True

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor], **kwargs):
        """!
        Construct a new operator sharing same kwargs with new inputs
        and outputs

        @param inputs List[IRTensor]: input tensors
        @param outputs List[IRTensor]: output tensors

        @return op IRDimop: the new constructed operator
        """
        op = self._create_fn[0](*inputs, **kwargs, signature=self.signature)
        # annos = self._annos_candidates
        # rules = self._trans_rules
        # op = IRDimops(self.signature, annos, inputs, self.name, rules, **kwargs)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op

    def align(self, signature, inputs: List[IRTensor], op_anno: OpAnno, kwargs: Dict) -> bool:
        """!
        Align input tensor shapes to the operator annotation.

        @param inputs List[IRTensor]: input tensor list
        @param op_anno OpAnno: operator annotation

        @return success True if align success else False
        """
        identifiers = op_anno.identifiers
        # input shape match
        if len(op_anno.inputs()) != len(inputs):
            return False
        # expand *
        expand_dims = None
        if '*' in identifiers:
            candicates = [c for c in string.ascii_lowercase if c not in identifiers]
            # go through inputs
            for idx, (ashape, itensor) in enumerate(zip(op_anno.inputs(), inputs)):
                names = [dim_anno.name for dim_anno in ashape.dims]
                if '*' in names:
                    if not isinstance(itensor, IRTensor):
                        return False
                    pos = names.index('*')
                    reduce = ashape[pos].reduces[0].value
                    ndims = len(inputs[idx].shape) - (len(names) - 1)
                    if expand_dims is not None and len(expand_dims) != ndims:
                        return False
                    if expand_dims is None:
                        expand_dims = []
                        if ndims > 0:
                            expand_dims = list(DimAnno(candicates[dim] + reduce) for dim in range(ndims))
                    shape_anno = list(op_anno.input(idx).dims[:pos]) + expand_dims + list(op_anno.input(idx).dims[pos+1:])
                    shape_anno = ShapeAnno(tuple(shape_anno))
                    op_anno.set_input(idx, shape_anno)
            # * should appear in inputs
            assert expand_dims is not None, f"Syntax Error: {op_anno}: * should also appear in inputs"
            # go through outputs
            for idx, shape_anno in enumerate(op_anno.outputs()):
                names = [dim_anno.name for dim_anno in shape_anno.dims]
                if '*' in names:
                    pos = names.index('*')
                    shape_anno = list(op_anno.output(idx).dims[:pos]) + expand_dims + list(op_anno.output(idx).dims[pos+1:])
                    shape_anno = ShapeAnno(tuple(shape_anno))
                    op_anno.set_output(idx, shape_anno)
            op_anno.reset_identifiers()

        identifier_values: Dict[str, int] = dict()
        for ashape, itensor in zip(op_anno.inputs(), inputs):
            if not isinstance(itensor, IRTensor) or ashape.ignore:
                continue
            if ashape.ndims != len(itensor.shape):
                return False
            for adim, dimlen in zip(ashape.dims, itensor.shape):
                if len(adim.identifiers) == 1:
                    if adim.identifiers[0] in identifier_values and identifier_values[adim.identifiers[0]] != dimlen:
                        raise RuntimeError(f'the exist identifier value {identifier_values[adim.identifiers[0]]} is not equal to the new value {dimlen}')
                    identifier_values[adim.identifiers[0]] = dimlen

        # check dimension consistency
        for ashape, itensor in zip(op_anno.inputs(), inputs):
            if itensor is None:
                continue
            if ashape.ignore:
                continue
            if not isinstance(itensor, IRTensor):
                continue
            if ashape.ndims != len(itensor.shape):
                return False
            for adim, dimlen in zip(ashape.dims, itensor.shape):
                ret = True
                identifiers = adim.identifiers
                if len(identifiers) == 1:
                    ret = op_anno.setlen(identifiers[0], dimlen)
                else:
                    toinfer, accum = [], 1
                    for identifier in identifiers:
                        length = op_anno.getlen(identifier)
                        if length is None:
                            if identifier in kwargs:
                                if isinstance(kwargs[identifier], IRObject):
                                    _logger.warning(
                                        f"Function {signature}: Found identifier {identifier} in kwargs to be IRObject, "
                                        f"this will turn it into a static value. Pay attention to the usage "
                                        f"in dynamic-shape scenarios")
                                    kwargs[identifier] = kwargs[identifier].value
                                length = kwargs[identifier]
                                if not isinstance(length, int):
                                    raise ValueError(
                                        f"Function {signature}: identifier {identifier} in kwargs "
                                        f"must be int or IRObject[value=int], but got {length}")
                                ret = op_anno.setlen(identifier, length)
                                accum *= length
                            elif identifier in identifier_values:
                                ret = op_anno.setlen(identifier, identifier_values[identifier])
                                accum *= identifier_values[identifier]
                            else:
                                toinfer.append(identifier)
                        else:
                            accum *= length
                    if len(toinfer) == 0 and accum != dimlen:
                        return False
                    assert len(toinfer) <= 1, f"Syntax Error {op_anno}: cannot infer hidden dim: {adim}"
                    if len(toinfer) == 1:
                        assert dimlen % accum == 0, f"{dimlen} % {accum} != 0"
                        ret = op_anno.setlen(toinfer[0], dimlen // accum)
                if not ret:
                    return False
        return True

    def algorithms(self, tag: Optional[str] = None):
        factory = DistAlgorithmFactory()
        if tag is None:
            algos = list()
            if factory.exist(type(self)):
                algos += [template(self) for template in factory.algorithms(type(self))]
            if factory.exist(IRDimops):
                algos += [template(self) for template in factory.algorithms(IRDimops)]
            return algos
        else:
            if factory.exist(type(self), tag):
                template = factory.algorithms(type(self), tag)
                return template(self)
            if factory.exist(IRDimops, tag):
                template = factory.algorithms(IRDimops, tag)
                return template(self)
            return None

    def transform_space(self) -> List[Tuple[int, int]]:
        """
        Get transformation space of the operator, the transformation space
        represents all configurations that can be segmented

        @return List[Tuple[int, int]]: list of (idx, dim)
        """
        return self.anno.transform_space()
