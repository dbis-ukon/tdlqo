from typing import Optional

import torch
import torch.nn as nn
import torch_scatter

class MultiHeadAggregation(torch.nn.Module):
    def __init__(self, input_size: int, output_size_per_head: int, num_heads: int, count: bool):
        super().__init__()
        self.num_heads = num_heads
        self.output_size_per_head = output_size_per_head
        self.count = count

        # Shared batched linear layers for all heads
        self.att_weight = nn.Linear(input_size, num_heads, bias=False)
        self.value_proj = nn.Linear(input_size, num_heads * output_size_per_head, bias=False)

        self._output_size = num_heads * output_size_per_head + (1 if count else 0)

    def output_size(self):
        return self._output_size

    def forward(self, x: torch.FloatTensor, index: torch.FloatTensor, indptr: torch.LongTensor, count: Optional[torch.FloatTensor] = None, dim_size: int = None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1

        att = self.att_weight(x)  # [N, H]
        val = self.value_proj(x)  # [N, H * D]
        val = val.view(-1, self.num_heads, self.output_size_per_head)  # [N, H, D]

        att_exp = att.exp()  # [N, H]

        att_sum = torch_scatter.segment_csr(att_exp, indptr, reduce='sum')  # [G, H]
        att_sum[att_sum == 0] = 1
        att_norm = att_exp / att_sum[index]

        weighted = att_norm.unsqueeze(-1) * val  # [N, H, D]
        out_val = torch_scatter.segment_csr(weighted, indptr, reduce='sum')  # [G, H, D]
        out = out_val.view(dim_size, -1)  # [G, H*D]

        if self.count:
            out = torch.cat([out, count], dim=1)

        return out
