from typing import FrozenSet, Optional, Dict

import queries.predicates.conjunction
import queries.predicates.disjunction
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.table_occurrence import TableOccurrence
from sample_encoders.table_row import TableRow
from schemas.column import Column


class Negation(Predicate):
    def __init__(self, predicate: Predicate):
        self._predicate = predicate
        not_pattern = "NOT (" + predicate.pattern() + ")"
        super().__init__(not_pattern, predicate.pattern_map())

    def _get_simple_hash(self) -> int:
        return hash(("NOT", self._predicate.simple_hash()))

    def predicate(self) -> Predicate:
        return self._predicate

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        return isinstance(other, Negation) and self._predicate.syntactically_equal(other.predicate(), table_occurrence_mapping)

    def _check_normal_form(self) -> bool:
        return isinstance(self._predicate, SimplePredicate)

    def _get_normal_form(self) -> Optional[Predicate]:
        if isinstance(self._predicate, Negation):
            return self._predicate.predicate().get_normal_form()
        elif isinstance(self._predicate, queries.predicates.conjunction.Conjunction):
            normal_form_predicates = []
            for predicate in self._predicate.predicates():
                normal_form_predicate = Negation(predicate).get_normal_form()
                if normal_form_predicate is None:
                    return None
                normal_form_predicates.append(normal_form_predicate)
            return queries.predicates.disjunction.Disjunction(normal_form_predicates).get_normal_form()

        elif isinstance(self._predicate, queries.predicates.disjunction.Disjunction):
            normal_form_predicates = []
            for predicate in self._predicate.predicates():
                normal_form_predicate = Negation(predicate).get_normal_form()
                if normal_form_predicate is None:
                    return None
                normal_form_predicates.append(normal_form_predicate)
            return queries.predicates.conjunction.Conjunction(normal_form_predicates).get_normal_form()
        else:
            raise NotImplementedError()

    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        result = self._predicate.evaluate(table_row)
        if result is None:
            return None
        return not result

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        return Negation(self._predicate.replace(replacements))

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        return self._predicate.columns_for(table_occurrence)

