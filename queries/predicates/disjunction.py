
from typing import FrozenSet, List, Optional, Dict

from frozenlist import FrozenList

import queries.predicates.conjunction
from queries.predicates.comparison_predicate import ComparisonPredicate
from queries.predicates.negation import Negation
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.table_occurrence import TableOccurrence
from sample_encoders.table_row import TableRow
from schemas.column import Column


class Disjunction(Predicate):
    def __init__(self, predicates: List[Predicate]):
        or_pattern = "(" + " OR ".join([predicate.pattern() for predicate in predicates]) + ")"
        pattern_map = []
        for predicate in predicates:
            pattern_map.extend(predicate.pattern_map())
        super().__init__(or_pattern, pattern_map)
        self._predicates = FrozenList(predicates)
        self._predicates.freeze()

    def predicates(self) -> FrozenList[Predicate]:
        return self._predicates

    def _get_simple_hash(self) -> int:
        return hash(("OR", frozenset(predicate.simple_hash() for predicate in self._predicates)))

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        if not isinstance(other, Disjunction):
            return False
        other_predicates = other.predicates()
        other_predicates_dict = {}
        for other_predicate in other_predicates:
            simple_hash = other_predicate.simple_hash()
            if simple_hash not in other_predicates_dict:
                other_predicates_dict[simple_hash] = []
            other_predicates_dict[simple_hash].append(other_predicate)
        checked_other_predicates = set()
        for predicate in self._predicates:
            simple_hash = predicate.simple_hash()
            found_matching_predicate = False
            for other_predicate in other_predicates_dict.get(simple_hash, []):
                if other_predicate not in checked_other_predicates and predicate.syntactically_equal(other_predicate, table_occurrence_mapping):
                    checked_other_predicates.add(other_predicate)
                    found_matching_predicate = True
            if not found_matching_predicate:
                return False
        for other_predicate in other_predicates:
            if other_predicate not in checked_other_predicates:
                return False
        return True

    def _check_normal_form(self) -> bool:
        return len(self._predicates) > 1 and all((isinstance(predicate, Negation) or isinstance(predicate, (SimplePredicate, ComparisonPredicate))) and predicate.check_normal_form() for predicate in self._predicates)

    def _get_normal_form(self) -> Optional[Predicate]:
        normal_form_predicates = [[]]
        for predicate in self._predicates:
            normal_form_predicate = predicate.get_normal_form()
            if normal_form_predicate is None:
                return None
            elif isinstance(normal_form_predicate, Disjunction):
                for sub_predicate in normal_form_predicate.predicates():
                    for normal_form_disjunction in normal_form_predicates:
                        normal_form_disjunction.append(sub_predicate)
            elif isinstance(normal_form_predicate, queries.predicates.conjunction.Conjunction):
                new_normal_form_predicates = []
                for sub_predicate in normal_form_predicate.predicates():
                    for normal_form_disjunction in normal_form_predicates:
                        new_normal_form_predicates.append(normal_form_disjunction + [sub_predicate])
                normal_form_predicates = new_normal_form_predicates
            else:
                for normal_form_disjunction in normal_form_predicates:
                    normal_form_disjunction.append(normal_form_predicate)
        if len(normal_form_predicates) == 1:
            return Disjunction(normal_form_predicates[0])
        else:
            return queries.predicates.conjunction.Conjunction([Disjunction(normal_form_disjunction) for normal_form_disjunction in normal_form_predicates])

    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        all_false = True
        for predicate in self._predicates:
            result = predicate.evaluate(table_row)
            if result is None:
                all_false = False
            elif result:
                return True
        return False if all_false else None

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        new_predicates = []
        for predicate in self._predicates:
            new_predicates.append(predicate.replace(replacements))
        return Disjunction(new_predicates)

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        columns = set()
        for predicate in self._predicates:
            columns.update(predicate.columns_for(table_occurrence))
        return frozenset(columns)



