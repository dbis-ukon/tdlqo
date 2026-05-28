
from __future__ import annotations

from enum import Enum
from typing import Optional

import torch


class PlanType(Enum):
    JOIN = 0
    SCAN = 1


class TopDownLearnedOptimizerTargetData:
    def __init__(self,
                 join_operator_min_cost: torch.FloatTensor,
                 join_operator_max_cost: torch.FloatTensor,
                 join_operator_cost_mask: torch.BoolTensor,
                 join_operator_subgroup_cost: Optional[torch.FloatTensor],
                 scan_operator_min_cost: torch.FloatTensor,
                 scan_operator_max_cost: torch.FloatTensor,
                 scan_operator_cost_mask: torch.BoolTensor):
        self.join_operator_min_cost = join_operator_min_cost
        self.join_operator_max_cost = join_operator_max_cost
        self.join_operator_cost_mask = join_operator_cost_mask
        self.join_operator_subgroup_cost = join_operator_subgroup_cost
        self.scan_operator_min_cost = scan_operator_min_cost
        self.scan_operator_max_cost = scan_operator_max_cost
        self.scan_operator_cost_mask = scan_operator_cost_mask

    @staticmethod
    def unify(target_data_list):
        join_operator_min_costs = []
        join_operator_max_costs = []
        join_operator_cost_masks = []
        join_operator_subgroup_costs = []
        scan_operator_min_costs = []
        scan_operator_max_costs = []
        scan_operator_cost_masks = []
        for target_data in target_data_list:
            join_operator_min_costs.append(target_data.join_operator_min_cost)
            join_operator_max_costs.append(target_data.join_operator_max_cost)
            join_operator_cost_masks.append(target_data.join_operator_cost_mask)
            if target_data.join_operator_subgroup_cost is not None:
                join_operator_subgroup_costs.append(target_data.join_operator_subgroup_cost)
            scan_operator_min_costs.append(target_data.scan_operator_min_cost)
            scan_operator_max_costs.append(target_data.scan_operator_max_cost)
            scan_operator_cost_masks.append(target_data.scan_operator_cost_mask)
        return TopDownLearnedOptimizerTargetData(torch.cat(join_operator_min_costs, dim=0),
                                                 torch.cat(join_operator_max_costs, dim=0),
                                                 torch.cat(join_operator_cost_masks, dim=0),
                                                 torch.cat(join_operator_subgroup_costs, dim=0) if len(join_operator_subgroup_costs) > 0 else None,
                                                 torch.cat(scan_operator_min_costs, dim=0),
                                                 torch.cat(scan_operator_max_costs, dim=0),
                                                 torch.cat(scan_operator_cost_masks, dim=0))


    def to(self, device: torch.device) -> TopDownLearnedOptimizerTargetData:
        return TopDownLearnedOptimizerTargetData(
            self.join_operator_min_cost.to(device),
            self.join_operator_max_cost.to(device),
            self.join_operator_cost_mask.to(device),
            self.join_operator_subgroup_cost.to(device) if self.join_operator_subgroup_cost is not None else None,
            self.scan_operator_min_cost.to(device),
            self.scan_operator_max_cost.to(device),
            self.scan_operator_cost_mask.to(device)
        )


