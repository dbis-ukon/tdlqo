from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityMode, CardinalityEstimator
from cost_models.cost_model import CostModel
from cost_models.local_cost_model import LocalCostModel
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class COuterCostModel(LocalCostModel):
    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        if isinstance(relational_algebra_expression, JoinExpression):
            inner_query = relational_algebra_expression.outer.canonical_query()
            return max(self.cardinality_estimator.estimate(inner_query), 1)
        else:
            return 1

    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        return COuterCostModel(self.cardinality_estimator)




