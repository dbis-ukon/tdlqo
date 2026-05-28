
from __future__ import annotations

from typing import Any, Optional, Dict, Set, List

from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence


class UnfilteredSPJQuery:
    def __init__(self, spj_query: SPJQuery):
        self._spj_query = spj_query
        self._hash = None


    def known_equal(self, other: UnfilteredSPJQuery) -> bool:
        return self.known_equal_assignment(other) is not None

    @staticmethod
    def _build_viable_assignments(table_assignment_candidates: Dict[TableOccurrence, List[TableOccurrence]],
                                  current_assignment: Dict[TableOccurrence, TableOccurrence],
                                  locked_other_occurrences: Set[TableOccurrence]) -> List[Dict[TableOccurrence, TableOccurrence]]:
        chosen_occurrence = None
        for table_occurrence in table_assignment_candidates:
            if table_occurrence not in current_assignment:
                chosen_occurrence = table_occurrence
                break
        if chosen_occurrence is None:
            return [current_assignment.copy()]
        viable_assignments = []
        for other_occurrence in table_assignment_candidates[chosen_occurrence]:
            if other_occurrence not in locked_other_occurrences:
                current_assignment[chosen_occurrence] = other_occurrence
                locked_other_occurrences.add(other_occurrence)
                viable_assignments.extend(UnfilteredSPJQuery._build_viable_assignments(table_assignment_candidates, current_assignment, locked_other_occurrences))
                del current_assignment[chosen_occurrence]
                locked_other_occurrences.remove(other_occurrence)
        return viable_assignments

    def known_equal_assignment(self, other: UnfilteredSPJQuery) -> Optional[Dict[TableOccurrence, TableOccurrence]]:
        own_table_occurrences = self._spj_query.table_occurrences_by_table()
        other_table_occurrences = other._spj_query.table_occurrences_by_table()
        if own_table_occurrences.keys() != other_table_occurrences.keys():
            return None
        for table in own_table_occurrences:
            if len(own_table_occurrences[table]) != len(other_table_occurrences[table]):
                return None
        assignment_candidates = {}
        for table in own_table_occurrences:
            table_assignment_candidates = {}
            for table_occurrence in own_table_occurrences[table]:
                table_assignment_candidates[table_occurrence] = list(other_table_occurrences[table])
            viable_assignments = UnfilteredSPJQuery._build_viable_assignments(table_assignment_candidates, {}, set())
            if len(viable_assignments) == 0:
                return None
            assignment_candidates[table] = viable_assignments
        full_assignments = [{}]
        for table in assignment_candidates:
            new_full_assignments = []
            for assignment in full_assignments:
                for new_assignment in assignment_candidates[table]:
                    new_full_assignments.append({**assignment, **new_assignment})
            full_assignments = new_full_assignments
        for assignment in full_assignments:
            if self._assignment_equality(other, assignment):
                return assignment
        return None

    def _assignment_equality(self, other: UnfilteredSPJQuery, table_occurrence_mapping: Dict[TableOccurrence, TableOccurrence]) -> bool:
        if len(self._spj_query.non_equi_join_predicates()) > 0 or len(other._spj_query.non_equi_join_predicates()) > 0:
            raise NotImplementedError()
        other_join_predicate = other._spj_query.join_predicate()
        return self._spj_query.join_predicate().semantically_equal(other_join_predicate, table_occurrence_mapping=table_occurrence_mapping)


    def __hash__(self):
        if self._hash is None:
            table_set = frozenset([hash(table_occurrence.table()) for table_occurrence in self._spj_query.table_occurrences()])
            join_set = frozenset([frozenset([(table_occurrence.table(), column) for table_occurrence, column in join.equivalence_class()]) for join in self._spj_query.joins()])
            assert len(self._spj_query.non_equi_join_predicates()) == 0
            self._hash = hash((table_set, join_set))
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnfilteredSPJQuery):
            return False
        elif self is other:
            return True
        else:
            return self.known_equal(other)

