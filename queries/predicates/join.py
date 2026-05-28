
from __future__ import annotations

from typing import Tuple, Iterable, Optional, Dict, FrozenSet, List

from queries.predicates.predicate import Predicate
from queries.table_occurrence import TableOccurrence
from schemas.column import Column
from schemas.table import Table


class Join(Predicate):
    def __init__(self, equivalence_class: Iterable[Tuple[TableOccurrence, Column]], redundant: bool = False):
        self._equivalence_class = frozenset(equivalence_class)
        self._redundant = redundant
        last_table_occurrence = None
        last_column = None
        pattern_components = []
        table_occurrences = []
        for i, (table_occurrence, column) in enumerate(self._equivalence_class):
            if last_table_occurrence is not None:
                pattern_components.append("%s." + column.name() + " = %s." + last_column.name())
                table_occurrences.append(table_occurrence)
                table_occurrences.append(last_table_occurrence)
            last_table_occurrence = table_occurrence
            last_column = column
        pattern = " AND ".join(pattern_components)
        super().__init__(pattern, table_occurrences)

    def equivalence_class(self) -> FrozenSet[Tuple[TableOccurrence, Column]]:
        return self._equivalence_class

    def is_redundant(self) -> bool:
        return self._redundant

    def set_redundant(self, redundant: bool = True) -> None:
        self._redundant = redundant

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        if not isinstance(other, Join):
            return False
        if table_occurrence_mapping is None:
            equivalence_class = self.equivalence_class()
        else:
            equivalence_class = []
            for table_occurrence, column in self.equivalence_class():
                equivalence_class.append((table_occurrence_mapping[table_occurrence], column))
            equivalence_class = frozenset(equivalence_class)
        return equivalence_class == other.equivalence_class()

    @staticmethod
    def merge_joins(joins: List[Join]) -> List[Join]:
        join_set = set()
        join_dict = {}
        for join in joins:
            merge_set = set()
            for column_key in join.equivalence_class():
                if column_key in join_dict:
                    merge_set.add(join_dict[column_key])
            if len(merge_set) == 0:
                new_join = join
            else:
                equivalence_class = list(join.equivalence_class())
                redundant = join.is_redundant()
                for other_join in merge_set:
                    equivalence_class += list(other_join.equivalence_class())
                    redundant = redundant or other_join.is_redundant()
                    join_set.remove(other_join)
                new_join = Join(equivalence_class, redundant=redundant)
            for column_key in new_join.equivalence_class():
                join_dict[column_key] = new_join
            join_set.add(new_join)
        return list(join_set)

    def sort_key(self) -> str:
        parts = sorted(t.name() + "." + c.name() for t, c in self.signature())
        return ",".join(parts)

    def signature(self) -> FrozenSet[Tuple[Table, Column]]:
        return frozenset((table_occurrence.table(), column) for table_occurrence, column in self._equivalence_class)

    def _get_simple_hash(self) -> int:
        return hash(self.signature())

    def _check_normal_form(self) -> bool:
        return True

    def _get_normal_form(self) -> Optional[Predicate]:
        return self

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        new_equivalence_class = []
        for table_occurrence, column in self._equivalence_class:
            new_table_occurrence = replacements.get(table_occurrence, table_occurrence)
            new_equivalence_class.append((new_table_occurrence, column))
        return Join(new_equivalence_class, redundant=self._redundant)

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        return frozenset(column for to, column in self._equivalence_class if to == table_occurrence)

