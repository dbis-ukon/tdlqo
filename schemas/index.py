from typing import Iterable, List, Tuple

from frozenlist import FrozenList

from schemas.column import Column


class Index:
    def __init__(self, name: str, columns: List[Column], unique: bool, level: int, pages: int, tuples: int, prefix_distinct_counts: List[int]):
        self._name = name
        self._columns = FrozenList(columns)
        self._columns.freeze()
        self._unique = unique
        self._level = level
        self._pages = pages
        self._tuples = tuples
        self._prefix_distinct_counts = prefix_distinct_counts

    def name(self) -> str:
        return self._name

    def columns(self) -> FrozenList[Column]:
        return self._columns

    def unique(self) -> bool:
        return self._unique

    def level(self) -> int:
        return self._level

    def pages(self) -> int:
        return self._pages

    def tuples(self) -> int:
        return self._tuples

    def prefix_distinct_count(self, prefix_length: int) -> int:
        return self._prefix_distinct_counts[prefix_length - 1]

    def covers_columns(self, columns: Iterable[Column]) -> bool:
        return set(columns).issubset(set(self._columns))
