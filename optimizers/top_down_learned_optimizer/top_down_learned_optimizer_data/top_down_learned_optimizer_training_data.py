
from __future__ import annotations

from typing import List

import torch
import torch_geometric.data

from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import TopDownLearnedOptimizerData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import TopDownLearnedOptimizerTargetData


class TopDownLearnedOptimizerTrainingData:
    def __init__(self, input_data: TopDownLearnedOptimizerData, target_data: List[TopDownLearnedOptimizerTargetData]):
        self.input_data = input_data
        self.target_data = target_data

    @staticmethod
    def unify(training_data_list: List[TopDownLearnedOptimizerTrainingData]) -> TopDownLearnedOptimizerTrainingData:
        input_data_list = []
        target_data_list = []
        for training_data in training_data_list:
            input_data_list.append(training_data.input_data)
            target_data_list.append(training_data.target_data)
        input_loader = torch_geometric.loader.DataLoader(input_data_list, batch_size=len(input_data_list))
        input_data = next(iter(input_loader))
        target_data = TopDownLearnedOptimizerTrainingData.unify_targets(target_data_list)
        return TopDownLearnedOptimizerTrainingData(input_data, target_data)

    def to(self, device: torch.device) -> TopDownLearnedOptimizerTrainingData:
        return TopDownLearnedOptimizerTrainingData(
            self.input_data.to(device),
            [target.to(device) for target in self.target_data]
        )

    @staticmethod
    def unify_targets(target_data_list: List[List[TopDownLearnedOptimizerTargetData]]) -> List[TopDownLearnedOptimizerTargetData]:
        unified_targets = []
        for i in range(len(target_data_list[0])):
            target_i = [target[i] for target in target_data_list]
            unified_targets.append(TopDownLearnedOptimizerTargetData.unify(target_i))
        return unified_targets



