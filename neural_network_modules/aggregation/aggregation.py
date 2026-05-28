from enum import Enum

import torch
import torch_scatter
from torch import FloatTensor, LongTensor
from torch.nn import Linear
from torch_geometric.nn import AttentionalAggregation, SumAggregation, MaxAggregation

from neural_network_modules.aggregation.resize_aggregation import ResizeAggregation


class AggregationType(Enum):
    ATTENTION = 1
    SUM = 2
    MAX = 3


def build_aggregation(aggregation_type: AggregationType, input_size: int, output_size: int) -> torch.nn.Module:
    if aggregation_type == AggregationType.ATTENTION:
        return AttentionalAggregation(Linear(input_size, 1), nn=Linear(input_size, output_size))
    elif aggregation_type == AggregationType.SUM:
        return ResizeAggregation(input_size, output_size, SumAggregation())
    elif aggregation_type == AggregationType.MAX:
        return ResizeAggregation(input_size, output_size, MaxAggregation())


def aggregate_many_to_many(input_tensor: FloatTensor, index: LongTensor, dim_size: int) -> FloatTensor:
    gathered = torch.index_select(input_tensor, 0, index[0])
    scattered = torch_scatter.scatter(gathered, index[1], dim=0, dim_size=dim_size)
    return scattered


