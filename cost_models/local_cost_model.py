from abc import abstractmethod
from typing import Optional, Type, Dict

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cost_models.cost_model import CostModel
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class LocalCostModel(CostModel):
    def __init__(self, cardinality_estimator: CardinalityEstimator):
        super().__init__(cardinality_estimator)

    def cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        cost = self.local_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        if cost is None:
            return None
        for child in relational_algebra_expression.children:
            child_cost = self.cost(child, cardinality_mode=cardinality_mode)
            if child_cost is None:
                return None
            cost += child_cost
        return cost

    def operator_type_cost(self, relational_algebra_expression: RelationalAlgebraExpression, operator_type: Type[RelationalAlgebraExpression], cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        if isinstance(relational_algebra_expression, operator_type):
            cost = self.local_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
            if cost is None:
                return None
        else:
            cost = 0.0
        for child in relational_algebra_expression.children:
            child_cost = self.operator_type_cost(child, operator_type, cardinality_mode=cardinality_mode)
            if child_cost is None:
                return None
            cost += child_cost
        return cost

    def sublan_costs(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Dict[RelationalAlgebraExpression, float]:
        subplan_costs = {}
        cost = self.cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        if cost is not None:
            subplan_costs[relational_algebra_expression] = cost
        for child in relational_algebra_expression.children:
            child_subplan_costs = self.sublan_costs(child, cardinality_mode=cardinality_mode)
            for subplan, subplan_cost in child_subplan_costs.items():
                subplan_costs[subplan] = subplan_cost
        return subplan_costs

    @abstractmethod
    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        pass


