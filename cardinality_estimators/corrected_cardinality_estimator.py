from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from queries.query import Query


class CorrectedCardinalityEstimator(CardinalityEstimator):
    def __init__(self, primary_estimator: CardinalityEstimator, correction_estimator: PropheticCardinalityEstimator):
        self._primary_estimator = primary_estimator
        self._correction_estimator = correction_estimator

    def estimate(self, query: Query, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        correction_range = self._correction_estimator.estimate_range(query)
        if correction_range.min_cardinality == correction_range.max_cardinality:
            return correction_range.min_cardinality
        estimate = self._primary_estimator.estimate(query, cardinality_mode)
        if estimate is None:
            return None
        if correction_range.max_cardinality is not None and estimate > correction_range.max_cardinality:
            return correction_range.max_cardinality
        if estimate < correction_range.min_cardinality:
            return correction_range.min_cardinality
        return estimate



