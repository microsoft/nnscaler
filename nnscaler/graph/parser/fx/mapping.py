
from typing import Callable, Union
from functools import partial

import nnscaler.graph.function as function
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.parser.register import CustomizedOps


class SignFx2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if signature in SignFx2Op.kOpMap:
            function = SignFx2Op.kOpMap[signature]
            return partial(function, signature=signature)
        if CustomizedOps.exist(signature):
            return CustomizedOps.map(signature)
        raise KeyError(f"{signature} is not supported yet")

    @staticmethod
    def exist(signature: str) -> bool:
        if signature in SignFx2Op.kOpMap:
            return True
        if CustomizedOps.exist(signature):
            return True
        return False

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'
    __fcntemplate = lambda name: f'torch._C._nn.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # torch nn module
    __tnmtemplate = lambda name: f'torch.nn.{name}'

    # torch.Tensor template
    __tttemplate = lambda name: f'torch.Tensor.{name}'

    # runtime template
    __rtemplate = lambda name: f'nnscaler.runtime.function.function.{name}'

    # einops
    __einopsize = lambda name: f'einops._torch_specific.{name}'

    # custom ops
    __customops = lambda name: f'examples.custom_ops.{name}'

    kOpMap = {
        # __tnmtemplate('Dropout'): function.nnDropout,
        __fcntemplate('linear'): function.Linear,
        __ftemplate('dropout') : function.Dropout,
        __ttemplate('sum'): function.Sum,
        __ttemplate('mean') : function.Mean,
        __ttemplate('outer'): function.Outer,
        __ttemplate('erf'): function.Erf,
        __ttemplate('abs'): function.Abs,
        __ttemplate('exp'): function.Exp,
        'math.exp': function.Exp,
        __ttemplate('sqrt'): function.Sqrt,
        'math.sqrt': function.Sqrt,
        __ttemplate('log'): function.Log,
        __ttemplate('svd'): function.SVD,
        __ttemplate('diag'): function.Diag,
        'math.log': function.Log,
        __ttemplate('rsqrt'): function.RSqrt,
        __ttemplate('clamp'): function.Clamp,
        __ttemplate('clamp_min'): function.ClampMin,
        __ttemplate('squeeze'): function.Squeeze,
        __ttemplate('unsqueeze'): function.Unsqueeze,
        __tttemplate('type_as'): function.TypeAs,
        __ttemplate('gather'): function.Gather,
        __ttemplate('ceil'): function.Ceil,
        __ttemplate('sign'): function.Sign,
        __ttemplate('triu'): function.Triu,
        __ttemplate('tril'): function.Tril,
        __ftemplate('relu'): function.ReLU,
        __ftemplate('silu'): function.SiLU,
        __fcntemplate('log_sigmoid'): function.LogSigmoid,
        __fcntemplate('gelu'): function.GeLU,
        __ttemplate('eq') : function.CompareEQ,
        '_operator.eq': function.CompareEQ,
        __ttemplate('ne') : function.CompareNE,
        '_operator.ne': function.CompareNE,
        __ttemplate('max'): function.Max,
        __ttemplate('min'): function.Min,
        __ttemplate('where'): function.Where,
        __ttemplate('nonzero'): function.Nonzero,
        __ttemplate('nan_to_num') : function.NanToNum,
        __tttemplate('type'): function.Type,
        __tttemplate('long'): function.Long,
        __tttemplate('int'): function.Int,
        __tttemplate('float'): function.Float,
        __tttemplate('bool'): function.Bool,
        __ttemplate('fill_'): function.Fill,
        __ttemplate('masked_fill'): function.MaskedFill,
        __tttemplate('masked_fill_'): function.MaskedFill,
        __ttemplate('cumsum'): function.CumSum,
        __ttemplate('sigmoid'): function.Sigmoid,
        __tttemplate('sigmoid'): function.Sigmoid,
        __ftemplate('sigmoid') : function.Sigmoid,
        __fcntemplate('sigmoid') : function.Sigmoid,
        __ttemplate('tanh'): function.Tanh,
        __ftemplate('softmax') : function.Softmax,
        __ttemplate('softmax'): function.Softmax,
        __ftemplate('log_softmax') : function.LogSoftmax,
        __ttemplate('bmm') : function.BatchLinear,
        __ttemplate('pow'): function.Pow,
        '_operator.pow': function.Pow,
        __ttemplate('baddbmm'): function.BMMAdd,
        __ttemplate('permute'): function.Permute,
        __ttemplate('transpose'): function.Transpose,
        __tttemplate('expand'): function.Expand,
        __tttemplate('expand_as'): function.ExpandAs,
        __ttemplate('arange'): function.Arange,
        __ttemplate('linspace'): function.Linspace,
        __ttemplate('detach'): function.Detach,
        __ttemplate('_shape_as_tensor'): function.ShapeAsTensor,
        __ttemplate('index_select'): function.IndexSelect,
        __ttemplate('finfo'): function.FInfo,
        __ttemplate('inverse'): function.Inverse,
        __ttemplate('bitwise_or'): function.BitwiseOr,
        '_operator.or_': function.BitwiseOr,
        __ttemplate('bitwise_not'): function.BitwiseOr,
        '_operator.invert': function.BitwiseNot,
        __ftemplate('embedding'): function.Embedding,
        'torch.functional.einsum': function.EinSum,
        __ftemplate('unfold'): function.Unfold,
        __ftemplate('nll_loss') : function.NLLLoss,
        __ftemplate('l1_loss') : function.L1Loss,
        __ttemplate('norm'): function.Norm,
        'torch.functional.norm': function.Norm,
        __ftemplate('layer_norm'): function.LayerNorm,
        __ftemplate('scaled_dot_product_attention'): function.ScaledDotProductAttention,
        __fcntemplate('scaled_dot_product_attention'): function.ScaledDotProductAttention,

        # ============== runtime function =================
        __tttemplate('size'): function.Size,
        __tttemplate('to'): function.To,
        __tttemplate('dim'): function.Dim,
        '_operator.getitem': function.GetItem,
        '_operator.setitem': function.SetItem,
        'builtins.getattr': function.GetAttr,
        'builtins.tuple': function.MakeTuple,
        'builtins.list': function.MakeList,
        'builtins.slice': function.MakeSlice,
        'builtins.len': function.Len,
        'builtins.dict.keys': function.Dictkeys,
        'builtins.dict.values': function.DictValues,
        'builtins.dict.items': function.DictItems,

        # # torch nn functional
        '_operator.matmul': function.Matmul,
        'torch.mm': function.Matmul,
        __ttemplate('matmul'): function.Matmul,
        #
        # __ftemplate('_pad'): function.Pad,
        __ftemplate('cross_entropy'): function.CrossEntropy,
        #
        # # creators
        __ttemplate('empty'): function.Empty,
        __ttemplate('zeros'): function.Zeros,
        __ttemplate('zeros_like'): function.ZerosLike,
        __ttemplate('ones'): function.Ones,
        __ttemplate('ones_like'): function.OnesLike,
        __ttemplate('tensor'): function.NewTensor,
        __ttemplate('full'): function.Full,
        __ttemplate('full_like'): function.FullLike,
        __ttemplate('rand'): function.Rand,
        __ttemplate('rand_like'): function.RandLike,
        __ttemplate('randn'): function.Randn,
        __ttemplate('randn_like'): function.RandnLike,
        __ttemplate('clone'): function.Clone,

        '_operator.is_': function.Is,
        '_operator.is_not': function.IsNot,
        __ttemplate('isnan'): function.IsNan,
        __ttemplate('isinf'): function.IsInf,
        __ttemplate('any'): function.TorchAny,
        __ttemplate('add') : function.Add,
        '_operator.add': function.Add,
        __ttemplate('addmm'): function.Addmm,
        '_operator.iadd': function.Add, # FIXME: may waste memory
        __ttemplate('sub') : function.Sub,
        '_operator.sub': function.Sub,
        __ttemplate('mul') : function.Mul,
        '_operator.mul': function.Mul,
        '_operator.imul': function.Mul, # FIXME: may waste memory
        __ttemplate('multiply') : function.Mul,
        '_operator.mod': function.Mod,

        __ttemplate('div') : function.Div,
        __ttemplate('true_divide'): function.Div,
        '_operator.truediv': function.Div,
        __ttemplate('floor_divide') : function.FloorDiv,
        '_operator.floordiv': function.FloorDiv,

        __ttemplate('neg'): function.Neg,
        '_operator.neg': function.Neg,
        #
        __ttemplate('gt'): function.CompareGT,
        '_operator.gt': function.CompareGT,
        __ttemplate('lt'): function.CompareLT,
        '_operator.lt': function.CompareLT,
        __ttemplate('ge'): function.CompareGE,
        '_operator.ge': function.CompareGE,
        __ttemplate('le'): function.CompareLE,
        '_operator.le': function.CompareLE,
        #
        __ttemplate('sin'): function.Sin,
        #
        __ttemplate('cos'): function.Cos,
        #
        __tttemplate('view'): function.View,
        __tttemplate('contiguous'): function.Contiguous,

        __ttemplate('reshape'): function.Reshape,
        
        __ttemplate('conv1d'): function.Conv1D,
        #
        # __ttemplate('conv2d'): function.Conv2D,
        #
        # __ttemplate('conv3d'): function.Conv3D,
        #
        # __ttemplate('pad'): function.Pad,
        #
        # __ttemplate('select'): function.Select,
        #
        # __ttemplate('slice'): function.Slice,
        #
        # #pytorch1.11
        # __ttemplate('select_scatter'): function.SelectScatter,
        #
        __tttemplate('repeat'): function.Repeat,
        __ttemplate('cat'): function.Cat,
        __ttemplate('stack'): function.Stack,
        __ttemplate('chunk'): function.Chunk,
        __ttemplate('flatten'): function.Flatten,
        # __ttemplate('roll'): function.Roll,
        #
        # __ttemplate('adaptive_avg_pool1d'): function.AdaptiveAvgPool1d,
        #
        # runtime functions
        __rtemplate('anchor'): function.GraphAnchor,
        __rtemplate('identity'): function.Identity,
        __rtemplate('multiref'): function.MultiRef,
        __rtemplate('accum'): function.Accum,
        __rtemplate('setitem'): function.SetItem,

        # #einops
        # __einopsize('apply_for_scriptable_torch'): function.ScriptEinOps,

        'torch.functional.split': function.Split,
        __ttemplate('split'): function.Split,
        __tttemplate('split'): function.Split,
        __ttemplate('topk'): function.Topk,
    }
