from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cost_models.cost_model import CostModel
from cost_models.local_cost_model import LocalCostModel
from queries.predicates.true_predicate import TruePredicate
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression


# How good is this cost model, really?
class MainMemoryCostModel(LocalCostModel):
    def __init__(self, cardinality_estimator: CardinalityEstimator, tau_parameter: float, lambda_parameter: float):
        super().__init__(cardinality_estimator)
        self._tau = tau_parameter
        self._lambda = lambda_parameter

    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        return MainMemoryCostModel(cardinality_estimator, self._tau, self._lambda)

    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        if isinstance(relational_algebra_expression, SequentialScanExpression):
            return self._tau * relational_algebra_expression.table_occurrence.table().cardinality()
        elif isinstance(relational_algebra_expression, IndexScanExpression):
            return 0
        elif isinstance(relational_algebra_expression, HashJoinExpression):
            cardinality = self.cardinality_estimator.estimate(relational_algebra_expression.canonical_query(), cardinality_mode=cardinality_mode)
            if cardinality is None:
                return None
            return cardinality
        elif isinstance(relational_algebra_expression, NestedLoopJoinExpression) and isinstance(relational_algebra_expression.inner, IndexScanExpression):
            left_cardinality = self.cardinality_estimator.estimate(relational_algebra_expression.outer.canonical_query(), cardinality_mode=cardinality_mode)
            if left_cardinality is None:
                return None
            right_base_table_occurrence = TableOccurrence(relational_algebra_expression.inner.table_occurrence.table())
            right_base_table_occurrence.set_predicate(TruePredicate())
            right_base_scan = ScanExpression(right_base_table_occurrence)
            join_expression = JoinExpression(relational_algebra_expression.outer, right_base_scan, relational_algebra_expression.join_condition)
            join_cardinality = self.cardinality_estimator.estimate(join_expression.canonical_query(), cardinality_mode=cardinality_mode)
            if join_cardinality is None:
                return None
            return self._lambda * left_cardinality * max(join_cardinality / left_cardinality, 1)
        else:
            return None

