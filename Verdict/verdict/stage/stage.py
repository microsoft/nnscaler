#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List

from verdict.graph import Node, Lineage


class Stage:

    def __init__(
        self,
        id: int,
        snodes: List[Node],
        pnodes: List[Node],
        input_lineages: List[Lineage],
        output_lineages: List[Lineage],
    ):
        self.id: int = id
        self.snodes: List[Node] = snodes
        self.pnodes: List[Node] = pnodes
        self.input_lineages: List[Lineage] = input_lineages
        self.output_lineages: List[Lineage] = output_lineages
