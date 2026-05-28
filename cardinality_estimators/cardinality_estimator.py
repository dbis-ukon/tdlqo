from abc import abstractmethod
from typing import Optional
from enum import Enum

from queries.query import Query


class CardinalityMode(Enum):
    MEAN = 1
    MIN = 2
    MAX = 3


class CardinalityEstimator:
    @abstractmethod
    def estimate(self, query: Query, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        pass
