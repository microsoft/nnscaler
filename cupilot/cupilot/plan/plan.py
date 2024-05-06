# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json

from cube.ir.operator import IRFwOperation
from cube.graph.graph import IRGraph

@dataclass
class StageSpec:
    # estimation
    est_latency: float # in milliseconds
    est_memory: float # in types
    # config
    tp_size: int
    dp_size: int
    # node.cid -> (idx, num) | None
    tp_spec: Dict[int, Optional[Tuple[int, int]]]
    # node.cid -> node.name
    names: Dict[int, str]

    def __repr__(self) -> str:
        dscp = ''
        for cid, strategy in self.tp_spec.items():
            strategy = 'Replicate' if strategy is None else f"idx={strategy[0]}, dim={strategy[1]}, num={self.tp_size}"
            dscp += f'  {self.names[cid]}({cid}): {strategy}\n'
        return dscp

    def to_dict(self) -> Dict:
        return {
            'est_latency': self.est_latency,
            'est_memory': self.est_memory,
            'tp_size': self.tp_size,
            'dp_size': self.dp_size,
            'tp_spec': self.tp_spec,
            'names': self.names
        }
    
    @staticmethod
    def from_dict(d: Dict):
        tp_spec = {int(cid): spec for cid, spec in d['tp_spec'].items()}
        names = {int(cid): name for cid, name in d['names'].items()}
        return StageSpec(
            est_latency=d['est_latency'],
            est_memory=d['est_memory'],
            tp_size=d['tp_size'],
            dp_size=d['dp_size'],
            tp_spec=tp_spec,
            names=names
        )


@dataclass
class ParallelSpec:
    stages: Tuple[StageSpec]
    est_latency: Optional[float] = None

    def save(self, filename: str):
        """
        Save plan into json file
        """
        with open(filename, 'w') as f:
            state = {
                'est_latency': self.est_latency,
                'stages': [s.to_dict() for s in self.stages]
            }
            json.dump(state, f)

    def getstate(self) -> str:
        """
        Get plan state as json string
        """
        state = {
            'est_latency': self.est_latency,
            'stages': [s.to_dict() for s in self.stages],
        }
        return json.dumps(state)

    @staticmethod
    def loadstate(state: str):
        """
        Load plan from json string
        """
        state = json.loads(state)
        return ParallelSpec(
            tuple(StageSpec.from_dict(s) for s in state['stages']),
            float(state['est_latency'])
        )

    @staticmethod
    def load(filename: str, check_graph_consistent: IRGraph = None):
        """
        Load plan from json file
        """
        with open(filename, 'r') as f:
            state = json.load(f)
            est_latency, stages = state['est_latency'], state['stages']
        spec = ParallelSpec(
            tuple(StageSpec.from_dict(s) for s in stages),
            float(est_latency)
        )
        if check_graph_consistent is not None:
            graph = check_graph_consistent
            cid2name = {n.cid: n.name for n in graph.select(ntype=IRFwOperation)}
            for stage in spec.stages:
                for cid, name in stage.names.items():
                    assert cid in cid2name, f'graph is not consistent with plan: node cid {cid}:{name} not found in graph'
                    assert cid2name[cid] == name, f'graph is not consistent with plan: cid {cid}:{name} name mismatch'
        return spec
    
    def str(self, nmicros: Optional[int] = None) -> str:
        dscp = f'nstages: {len(self.stages)} | per-micro-batch latency: {round(self.est_latency, 2)} ms'
        if nmicros is not None:
            dscp += f' | e2e latency: {round(self.est_latency * nmicros / 1000, 3)} s' 
        for sidx, stage in enumerate(self.stages):
            tp, dp = stage.tp_size, stage.dp_size
            latency, memory = stage.est_latency, stage.est_memory / 1024 / 1024 / 1024
            dscp += f'\nStage {sidx} (tp={tp}, dp={dp}, '\
                    f'latency={round(latency, 2)} ms, memory={round(memory, 2)} GB):\n'
            dscp += f'{stage}'
        return dscp

    def __repr__(self) -> str:
        return self.str()
