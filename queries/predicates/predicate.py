from __future__ import annotations

from abc import abstractmethod
from typing import FrozenSet, Iterable, Optional, Dict

import queries.predicates.expression
import queries.table_occurrence
from sample_encoders.table_row import TableRow
from schemas.column import Column


class Predicate(queries.predicates.expression.Expression):
    def __init__(self, pattern: str, pattern_map: Iterable[queries.table_occurrence.TableOccurrence]):
        super().__init__(pattern, pattern_map)
        self._is_normal_form = None
        self._normal_form = None

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[queries.table_occurrence.TableOccurrence, queries.table_occurrence.TableOccurrence]] = None) -> bool:
        return False

    def semantically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[queries.table_occurrence.TableOccurrence, queries.table_occurrence.TableOccurrence]] = None) -> bool:
        normal_form = self.get_normal_form()
        if normal_form is None:
            return False
        other_normal_form = other.get_normal_form()
        if other_normal_form is None:
            return False
        return normal_form.syntactically_equal(other_normal_form, table_occurrence_mapping)

    def check_normal_form(self) -> bool:
        if self._is_normal_form is None:
            self._is_normal_form = self._check_normal_form()
            if self._is_normal_form:
                self._normal_form = self
        return self._is_normal_form

    def _check_normal_form(self) -> bool:
        return False

    def get_normal_form(self) -> Optional[Predicate]:
        if self._normal_form is None:
            if self._is_normal_form is None and self.check_normal_form():
                self._normal_form = self
            else:
                self._normal_form = self._get_normal_form()
        return self._normal_form

    def _get_normal_form(self) -> Optional[Predicate]:
        return None

    @abstractmethod
    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        raise NotImplementedError

    @abstractmethod
    def columns_for(self, table_occurrence: queries.table_occurrence.TableOccurrence) -> FrozenSet[Column]:
        raise NotImplementedError

    def replace(self, replacements: Dict[queries.table_occurrence.TableOccurrence, queries.table_occurrence.TableOccurrence]) -> Predicate:
        new_pattern_map = [replacements.get(to, to) for to in self.pattern_map()]
        return Predicate(self.pattern(), new_pattern_map)

