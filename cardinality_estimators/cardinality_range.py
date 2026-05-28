from __future__ import annotations
from typing import Optional


class CardinalityRange:
    def __init__(self, min_cardinality: int, max_cardinality: Optional[int]):
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    def merge(self, other: CardinalityRange) -> CardinalityRange:
        new_min = max(self.min_cardinality, other.min_cardinality)
        max_cardinalities = [cardinality for cardinality in [self.max_cardinality, other.max_cardinality] if cardinality is not None]
        if len(max_cardinalities) == 0:
            new_max = None
        else:
            new_max = min(max_cardinalities)
            if new_max < new_min:
                raise ValueError(f"Impossible cardinality range: min {new_min} > max {new_max}")
        return CardinalityRange(new_min, new_max)

    def is_exact(self) -> bool:
        return self.max_cardinality is not None and self.min_cardinality == self.max_cardinality

    def has_no_information(self) -> bool:
        return self.min_cardinality == 0 and self.max_cardinality is None

    @staticmethod
    def no_information_cardinality_range() -> CardinalityRange:
        return CardinalityRange(0, None)

    @staticmethod
    def exact_cardinality_range(cardinality: int) -> CardinalityRange:
        return CardinalityRange(cardinality, cardinality)

