from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityMode
from queries.query import Query


class AisheCardinalityEstimator:
    def __init__(self):
        pass

    def estimate(self, query: Query, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        return 165


