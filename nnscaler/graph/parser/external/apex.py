import copy
import logging
import string

from nnscaler.graph.function.dimops import ShapeAnno, OpAnno
from nnscaler.graph import parser

_logger = logging.getLogger(__name__)


try:

    from apex.normalization.fused_layer_norm import FusedLayerNormFunction, FusedLayerNormAffineFunction, FusedRMSNormFunction, FusedRMSNormAffineFunction

    def apex_fused_layer_norm_anno(input, normalized_shape, *args, **kwargs):
        """
        apex.normalization.fused_layer_norm.FusedLayerNormFunction
        """
        letters = iter(string.ascii_lowercase)
        input_anno = ShapeAnno.create_shape_str(input.shape, iterator=letters)
        ndims = len(input.shape)
        for dim in range(len(normalized_shape)):
            input_anno[ndims-1-dim] += '^'
        inputs = [input_anno, '?'] + ['?' for _ in args]
        outputs = [copy.copy(input_anno),]
        assert len(kwargs) == 0, f'torch.autgrad.Function receives unexpected kwargs ({kwargs}) for apply.'
        return OpAnno.create_op_str(inputs, outputs)


    # apex.normalization.fused_layer_norm.FusedRMSNormFunction
    apex_fused_rms_norm_anno = apex_fused_layer_norm_anno


    def apex_fused_layer_norm_affine_anno(input, weight, bias, normalized_shape, eps, *args, **kwargs) -> str:
        """
        apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction
        """
        assert not (weight is None and bias is not None), f"Not support for None of weight and parameter of bias"
        letters = iter(string.ascii_lowercase)
        anno_input = ShapeAnno.create_shape_str(input.shape, iterator=letters)
        ndims = len(input.shape)
        for dim in range(len(normalized_shape)):
            anno_input[ndims-1-dim] += '^'
        outputs = [copy.copy(anno_input),]
        inputs = [anno_input]
        inputs.append(ShapeAnno.create_shape_str(weight.shape, reduction='^', iterator=letters) if weight is not None else '?')
        inputs.append(ShapeAnno.create_shape_str(bias.shape, reduction='^', iterator=letters) if bias is not None else '?')
        inputs += ['?', '?']
        inputs += ['?' for _ in args]
        return OpAnno.create_op_str(inputs, outputs)


    def apex_fused_rms_norm_affine_anno(input, weight, normalized_shape, eps, *args, **kwargs) -> str:
        """
        apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction
        """
        letters = iter(string.ascii_lowercase)
        input_anno = ShapeAnno.create_shape_str(input.shape, iterator=letters)
        ndims = len(input.shape)
        for dim in range(len(normalized_shape)):
            input_anno[ndims-1-dim] += '^'
        outputs = [copy.copy(input_anno),]
        inputs = [input_anno]
        inputs.append(ShapeAnno.create_shape_str(weight.shape, reduction='^', iterator=letters) if weight is not None else '?')
        inputs += ['?', '?']
        inputs += ['?' for _ in args]
        return OpAnno.create_op_str(inputs, outputs)


    parser.register(apex_fused_layer_norm_anno)(FusedLayerNormFunction.apply)
    parser.register(apex_fused_layer_norm_affine_anno)(FusedLayerNormAffineFunction.apply)
    parser.register(apex_fused_rms_norm_anno)(FusedRMSNormFunction.apply)
    parser.register(apex_fused_rms_norm_affine_anno)(FusedRMSNormAffineFunction.apply)

except:
    _logger.warning('skip apex ops as it is not installed.')
