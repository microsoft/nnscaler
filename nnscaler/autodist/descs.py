from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import json
import copy
import yaml


@dataclass
class NodePartitionDesc:
    # list element: ((idx, dim), num), the order matters
    desc: List[Tuple[Tuple[int, int], int]]


@dataclass
class MeshDesc:
    # inter node
    row: int
    # intra node
    col: int

    @property
    def ngpus(self):
        return self.row * self.col

    def to_json(self):
        return (self.row, self.col)

    @staticmethod
    def from_json(val):
        return MeshDesc(*val)


@dataclass
class TensorParallelDesc:
    partition_descs: Dict[int, NodePartitionDesc]
    recompute_groups: List[List[int]]
    mesh_desc: MeshDesc
    analysis: Dict[str, Any]

    def to_json(self):
        ret = {}
        descs_list = [(k, v.desc) for k, v in self.partition_descs.items()]
        ret['partition_descs'] = descs_list
        ret['recompute_groups'] = self.recompute_groups
        ret['mesh_desc'] = self.mesh_desc.to_json()
        ret['analysis'] = self.analysis
        return ret

    @staticmethod
    def from_json(ret):
        partition_descs = {}
        for k, v in ret['partition_descs']:
            partition_descs[k] = NodePartitionDesc(v)
        return TensorParallelDesc(partition_descs,
                                  copy.deepcopy(ret['recompute_groups']),
                                  MeshDesc.from_json(ret['mesh_desc']),
                                  ret['analysis'])


@dataclass
class SPMDSearchOutput:
    desc: TensorParallelDesc
    memory: float
    all_time: float
    comp_time: float

    def to_json(self):
        return {
            'desc': self.desc.to_json(),
            'memory': self.memory,
            'all_time': self.all_time,
            'comp_time': self.comp_time,
        }

    @staticmethod
    def from_json(json_val):
        desc = TensorParallelDesc.from_json(json_val['desc'])
        return SPMDSearchOutput(desc, json_val['memory'], json_val['all_time'],
                                json_val['comp_time'])


@dataclass
class PipelineParallelDesc:
    spmd_descs: List[TensorParallelDesc]
    recompute_groups: List[List[int]]
    mesh_desc: MeshDesc

    def to_json(self):
        return {
            'spmd_descs': [desc.to_json() for desc in self.spmd_descs],
            'recompute_groups': self.recompute_groups,
            'mesh_desc': self.mesh_desc.to_json(),
        }

    @staticmethod
    def from_json(json_val):
        spmd_descs = []
        for spmd_desc_json in json_val['spmd_descs']:
            spmd_descs.append(TensorParallelDesc.from_json(spmd_desc_json))
        recompute_groups = copy.deepcopy(json_val['recompute_groups'])
        mesh_desc = MeshDesc.from_json(json_val['mesh_desc'])
        return PipelineParallelDesc(spmd_descs, recompute_groups, mesh_desc)


@dataclass
class PipelineSearchOutput:
    desc: PipelineParallelDesc
    e2e_time: float
    stage_mems: List[float]
    stage_all_times: List[float]
    stage_comp_times: List[float]

    def to_json(self):
        return {
            'desc': self.desc.to_json(),
            'e2e_time': self.e2e_time,
            'stage_mems': self.stage_mems,
            'stage_all_times': self.stage_all_times,
            'stage_comp_times': self.stage_comp_times,
        }

    @staticmethod
    def from_json(json_val):
        desc = PipelineParallelDesc.from_json(json_val['desc'])
        return PipelineSearchOutput(desc, json_val['e2e_time'],
                                    json_val['stage_mems'],
                                    json_val['stage_all_times'],
                                    json_val['stage_comp_times'])


@dataclass
class PartitionConstraint:

    # the name of the corresponding operator in the model. It equals
    # to the `signature` field in the `IRFwOperation` in cube
    name: str
    # the **closest** father module name of the operator
    parent_module: str
    # a list of allowed partition dimensions of input tensors
    allowed_partition_dims: List[Tuple[int, int]]
    replica_allowed: bool = True

    @staticmethod
    def from_json(content: Dict[str, Any]):
        allowed_partition_dims = [
            tuple(x) for x in content['allowed_partition_dims']
        ]
        return PartitionConstraint(content['name'], content['parent_module'],
                                   allowed_partition_dims,
                                   content['replica_allowed'])

    def to_json(self):
        return {
            'name': self.name,
            'parent_module': self.parent_module,
            'allowed_partition_dims': self.allowed_partition_dims,
            'replica_allowed': self.replica_allowed,
        }

    @staticmethod
    def from_yaml(content: Dict[str, Any]):

        def _parse_dims(dims: str) -> List[int]:
            return tuple([int(x) for x in dims.split(',')])

        allowed_partition_dims = [
            _parse_dims(x) for x in content['allowed_partition_dims']
        ]
        return PartitionConstraint(content['name'], content['parent_module'],
                                   allowed_partition_dims,
                                   content['replica_allowed'])

    def to_yaml(self):

        def to_str(dims: List[int]) -> str:
            return ','.join([str(x) for x in dims])

        allowed_partition_dims = [
            to_str(x) for x in self.allowed_partition_dims
        ]
        return {
            'name': self.name,
            'parent_module': self.parent_module,
            'allowed_partition_dims': allowed_partition_dims,
            'replica_allowed': self.replica_allowed,
        }

    def __hash__(self):
        return hash((self.name, self.parent_module,
                     tuple(self.allowed_partition_dims), self.replica_allowed))
