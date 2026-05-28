

from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cost_models.cost_model import CostModel
from cost_models.local_cost_model import LocalCostModel
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class COutCostModel(LocalCostModel):
    def __init__(self, cardinality_estimator: CardinalityEstimator):
        super().__init__(cardinality_estimator)

    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        cardinality = self.cardinality_estimator.estimate(relational_algebra_expression.canonical_query(), cardinality_mode=cardinality_mode)
        if cardinality is None:
            return None
        return max(1, cardinality)

    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        return COutCostModel(cardinality_estimator)



