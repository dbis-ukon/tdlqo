
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityEstimator
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class CostModel:
    def __init__(self, cardinality_estimator: CardinalityEstimator):
        self.cardinality_estimator = cardinality_estimator

    @abstractmethod
    def cost(self, relational_algebra_expression: RelationalAlgebraExpression) -> Optional[float]:
        pass

    @abstractmethod
    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        pass

