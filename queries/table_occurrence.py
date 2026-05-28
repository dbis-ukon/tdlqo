

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from schemas.table import Table

if TYPE_CHECKING:
    from queries.predicates.predicate import Predicate


class TableOccurrence:
    def __init__(self, table: Table, alias: Optional[str] = None):
        self._table = table
        self._predicate = None
        self._alias = alias
        self._simple_hash = None

    def table(self) -> Table:
        return self._table

    def predicate(self) -> Predicate:
        assert self._predicate is not None
        return self._predicate

    def alias(self) -> Optional[str]:
        return self._alias

    def set_predicate(self, predicate: Predicate):
        self._predicate = predicate
        self._simple_hash = None

    def known_equal(self, other: TableOccurrence) -> bool:
        assert self._predicate is not None
        return self._table == other.table() and self._predicate.semantically_equal(other.predicate(), {other: self})

    def sort_key(self) -> str:
        return self._table.name() + "\0" + (self._alias or "")

    def simple_hash(self) -> int:
        if self._simple_hash is None:
            assert self._predicate is not None
            predicate = self._predicate.get_normal_form()
            if predicate is None:
                predicate = self._predicate
            self._simple_hash = hash((self._table, predicate.simple_hash()))
        return self._simple_hash
