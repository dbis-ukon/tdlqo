from typing import Dict, Optional, Tuple

import torch

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cost_models.cost_model import CostModel
from cost_models.local_cost_model import LocalCostModel
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import \
    PlanType
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_encoder import TopDownLearnedOptimizerEncoder
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_module import TopDownLearnedOptimizerModule
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements
from schemas.schema import Schema


class TDLQOLearnedCostModel(LocalCostModel):
    """Local cost model backed by a trained TDLQO module loaded from a checkpoint.

    Predicts the local cost of a single operator without the forward-looking subgroup
    cost estimates that TDLQO adds during its greedy top-down search, so plan costs
    are obtained by summing local predictions over the operators of a plan."""

    def __init__(self, configuration: TopDownLearnedOptimizerConfiguration, schema: Schema, path: str):
        super().__init__(None)
        # The encoder must be loaded first: its key column vocabulary determines the module's edge layer size.
        self._encoder = TopDownLearnedOptimizerEncoder(configuration, schema)
        self._encoder.load(path)
        self._module = TopDownLearnedOptimizerModule(configuration, self._encoder)
        self._module.load_state_dict(torch.load(path + "_module.pt", map_location=torch.device("cpu")))
        self._table_occurrence_encodings: Dict[TableOccurrence, Tuple[torch.FloatTensor, torch.FloatTensor]] = {}

    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        query = relational_algebra_expression.query()
        assert isinstance(query, SPJQuery)
        self._encoder.add_table_occurrence_encodings(query, self._table_occurrence_encodings)
        group = GroupRelationalAlgebraExpression(query, Requirements())
        query_data, plan_dict, _ = self._encoder.encode(group, [relational_algebra_expression], table_occurrence_encodings=self._table_occurrence_encodings)
        self._module.adapt_join_information_size(self._encoder.join_information_size())
        with torch.no_grad():
            join_costs, scan_costs = self._module.local_forward(query_data)
        if (PlanType.JOIN, 0) in plan_dict:
            return float(join_costs[0])
        return float(scan_costs[0])

    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        raise NotImplementedError()
