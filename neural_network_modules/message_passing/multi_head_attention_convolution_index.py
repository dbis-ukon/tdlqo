from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.typing import Adj

from neural_network_modules.aggregation.multi_head_aggregation import MultiHeadAggregation


class MultiHeadAttentionConvolutionIndex(Module):
    def __init__(self,
                 in_channels: int,
                 edge_channels: int,
                 out_channels: int,
                 pre_size: int,
                 num_heads: int,
                 head_size: int,
                 count: bool,
                 activation_function: torch.nn.Module):
        super().__init__()
        self._aggr = MultiHeadAggregation(pre_size, head_size, num_heads, count)
        self._pre_layer = Linear(in_channels + edge_channels, pre_size)
        self._out_layer = Linear(in_channels + self._aggr.output_size(), out_channels)
        self._activation_function = activation_function

    def forward(self,
                x: torch.FloatTensor,
                edge_index: Adj,
                edge_indptr: torch.LongTensor,
                edge_count: Optional[torch.FloatTensor],
                edge_attr: torch.FloatTensor) -> Tensor:
        x_j = torch.index_select(x, 0, edge_index[0])
        message = torch.cat([x_j, edge_attr], dim=-1)
        message = self._pre_layer(message)
        message = self._activation_function(message)
        messages = self._aggr(message, edge_index[1], edge_indptr, count=edge_count, dim_size=x.size()[0])
        out = torch.cat([x, messages], dim=-1)
        out = self._out_layer(out)
        out = self._activation_function(out)
        return out
