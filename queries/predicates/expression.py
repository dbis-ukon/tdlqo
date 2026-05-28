from abc import abstractmethod
from typing import Iterable, FrozenSet

from frozenlist import FrozenList

from queries.table_occurrence import TableOccurrence


class Expression:
    def __init__(self, pattern: str, pattern_map: Iterable[TableOccurrence]):
        self._pattern = pattern
        self._table_occurrences = frozenset(pattern_map)
        self._pattern_map = FrozenList(pattern_map)
        self._pattern_map.freeze()
        self._simple_hash = None

    def table_occurrences(self) -> FrozenSet[TableOccurrence]:
        return self._table_occurrences

    def pattern_map(self) -> FrozenList[TableOccurrence]:
        return self._pattern_map

    def pattern(self) -> str:
        return self._pattern

    def alias_string(self) -> str:
        alias_tuple = tuple(table_occurrence.alias() for table_occurrence in self._pattern_map)
        return self._pattern % alias_tuple

    def simple_hash(self) -> int:
        if self._simple_hash is None:
            self._simple_hash = self._get_simple_hash()
        return self._simple_hash

    def _get_simple_hash(self) -> int:
        return hash((self._pattern, tuple(table_occurrence.table() for table_occurrence in self._pattern_map)))

