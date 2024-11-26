#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.graph.parser.parser import FxModuleParser
from nnscaler.graph.parser.converter import convert_model, to_fx_graph, to_ir_graph
from nnscaler.graph.parser.register import register
from nnscaler.graph.parser.external import *
