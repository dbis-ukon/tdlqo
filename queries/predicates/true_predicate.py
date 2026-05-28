from typing import FrozenSet, Optional, Dict

from queries.predicates.predicate import Predicate
from queries.table_occurrence import TableOccurrence
from sample_encoders.table_row import TableRow
from schemas.column import Column


class TruePredicate(Predicate):
    def __init__(self):
        super().__init__("TRUE", [])

    def _get_simple_hash(self) -> int:
        return hash(True)

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        return isinstance(other, TruePredicate)

    def _check_normal_form(self) -> bool:
        return True

    def _get_normal_form(self) -> Optional[Predicate]:
        return self

    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        return True

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        return self

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        return frozenset()





