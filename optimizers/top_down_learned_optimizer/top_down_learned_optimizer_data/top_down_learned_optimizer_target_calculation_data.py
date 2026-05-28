
from __future__ import annotations

from typing import Optional, List

import torch

from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import TopDownLearnedOptimizerData

class TopDownLearnedOptimizerTargetCalculationCostData:
    def __init__(self,
                 join_operator_min_cost: torch.FloatTensor,
                 join_operator_max_cost: torch.FloatTensor,
                 join_operator_cost_mask: torch.BoolTensor,
                 join_operator_outer_index: torch.LongTensor,
                 join_operator_inner_index: torch.LongTensor,
                 join_operator_subgroup_index: Optional[torch.FloatTensor],
                 scan_operator_min_cost: torch.FloatTensor,
                 scan_operator_max_cost: torch.FloatTensor,
                 scan_operator_cost_mask: torch.BoolTensor):
        self.join_operator_min_cost = join_operator_min_cost
        self.join_operator_max_cost = join_operator_max_cost
        self.join_operator_cost_mask = join_operator_cost_mask
        self.join_operator_outer_index = join_operator_outer_index
        self.join_operator_inner_index = join_operator_inner_index
        self.join_operator_subgroup_index = join_operator_subgroup_index
        self.scan_operator_min_cost = scan_operator_min_cost
        self.scan_operator_max_cost = scan_operator_max_cost
        self.scan_operator_cost_mask = scan_operator_cost_mask

    def to(self, device: torch.device) -> TopDownLearnedOptimizerTargetCalculationCostData:
        return TopDownLearnedOptimizerTargetCalculationCostData(
            self.join_operator_min_cost.to(device),
            self.join_operator_max_cost.to(device),
            self.join_operator_cost_mask.to(device),
            self.join_operator_outer_index.to(device),
            self.join_operator_inner_index.to(device),
            self.join_operator_subgroup_index.to(device) if self.join_operator_subgroup_index is not None else None,
            self.scan_operator_min_cost.to(device),
            self.scan_operator_max_cost.to(device),
            self.scan_operator_cost_mask.to(device)
        )


class TopDownLearnedOptimizerTargetCalculationData:
    def __init__(self,
                 targets: List[TopDownLearnedOptimizerData],
                 fixed_subgroup_costs: torch.FloatTensor,
                 cost_datas: List[TopDownLearnedOptimizerTargetCalculationCostData]):
        self.targets = targets
        self.fixed_subgroup_costs = fixed_subgroup_costs
        self.cost_datas = cost_datas

    def to(self, device: torch.device) -> TopDownLearnedOptimizerTargetCalculationData:
        return TopDownLearnedOptimizerTargetCalculationData(
            [target.to(device) for target in self.targets],
            self.fixed_subgroup_costs.to(device),
            [cost_data.to(device) for cost_data in self.cost_datas]
        )


