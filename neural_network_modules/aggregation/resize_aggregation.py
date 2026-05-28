from typing import Optional

from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import Aggregation


class ResizeAggregation(Aggregation):
    def __init__(self, input_size: int, output_size: int, aggregator: Aggregation):
        super(ResizeAggregation, self).__init__()
        self._resize = Linear(input_size, output_size)
        self._aggregator = aggregator

    def forward(self,
                x: Tensor,
                index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x = self._resize(x)
        return self._aggregator.forward(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)


