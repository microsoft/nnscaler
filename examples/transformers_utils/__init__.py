#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from packaging import version  
import transformers

from .causal_lm_wrapper import WrapperModel, aggregate_outputs_fn
from .tokenizer import get_tokenizer

if version.parse(transformers.__version__) >= version.parse('4.43.0'):    
    from .flash_attn_anno import *
else:
    # need specified support for each model if transformers version < 4.43.0
    pass
