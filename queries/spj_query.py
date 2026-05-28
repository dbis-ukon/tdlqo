
from __future__ import annotations

from typing import List, FrozenSet, Dict, Set, Tuple, Iterable, Optional, Any
from queries.predicates.join import Join
from queries.predicates.conjunction import Conjunction
from queries.predicates.predicate import Predicate
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.table_occurrence import TableOccurrence
from schemas.column import Column
from schemas.table import Table


class SelectClauseAggregationType:
    NONE = 0
    MIN = 1
    MAX = 2
    COUNT = 3


class SelectClauseElement:
    def __init__(self, table_occurrence: Optional[TableOccurrence], column: Optional[Column], aggregation_type: SelectClauseAggregationType):
        self.table_occurrence = table_occurrence
        self.column = column
        self.aggregation_type = aggregation_type


class SPJQuery(Query):
    def __init__(self, table_occurrences: Iterable[TableOccurrence], joins: Iterable[Join], non_equi_join_predicates: Iterable[Predicate], select_clause_elements: Optional[List[SelectClauseElement]] = None):
        # TODO: sorting, projections, deduplication, and basic aggregation
        self._table_occurrences = frozenset(table_occurrences)
        self._joins = frozenset(joins)
        self._join_predicate = Conjunction(list(joins))
        self._non_equi_join_predicates = frozenset(non_equi_join_predicates)
        if select_clause_elements is None:
            select_clause_elements = [SelectClauseElement(None, None, SelectClauseAggregationType.COUNT)]
        self._select_clause_elements = select_clause_elements
        super().__init__(None)
        self._hash = None
        self._join_graph_edge_dict: Optional[Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]]] = None
        self.query_name: Optional[str] = None

    def table_occurrences(self) -> FrozenSet[TableOccurrence]:
        return self._table_occurrences

    def table_occurrences_by_table(self) -> Dict[Table, Set[TableOccurrence]]:
        table_occurrences_by_table = {}
        for table_occurrence in self._table_occurrences:
            table = table_occurrence.table()
            if table not in table_occurrences_by_table:
                table_occurrences_by_table[table] = set()
            table_occurrences_by_table[table].add(table_occurrence)
        return table_occurrences_by_table

    def table_occurrence_by_alias(self, alias: str) -> Optional[TableOccurrence]:
        for table_occurrence in self._table_occurrences:
            if table_occurrence.alias() == alias:
                return table_occurrence
        return None

    def width(self) -> int:
        # TODO: account for projections
        return sum(table_occurrence.table().width() for table_occurrence in self._table_occurrences)

    def joins(self) -> FrozenSet[Join]:
        return self._joins

    def join_predicate(self) -> Conjunction:
        return self._join_predicate

    def non_equi_join_predicates(self) -> FrozenSet[Predicate]:
        return self._non_equi_join_predicates

    def select_clause_elements(self) -> List[SelectClauseElement]:
        return self._select_clause_elements

    def get_query_text(self, forced_select_clause: Optional[str]) -> str:
        return self._get_query_text(forced_select_clause=forced_select_clause)

    def _get_query_text(self, forced_select_clause: Optional[str] = None) -> str:
        if len(list(self._non_equi_join_predicates)) > 0:
            raise NotImplementedError()
        from_clause_parts = []
        where_clause_predicates = []
        for table_occurrence, joins in self._traverse():
            from_clause_part = "%s AS %s" % (table_occurrence.table().name(), table_occurrence.alias())
            if len(joins) > 0:
                from_clause_part += " ON %s" % " AND ".join([join.alias_string() for join in joins])
            from_clause_parts.append(from_clause_part)
            table_occurrence_predicate = table_occurrence.predicate()
            if not isinstance(table_occurrence_predicate, TruePredicate):
                where_clause_predicates.append(table_occurrence.predicate())
        from_clause = "\nJOIN ".join(from_clause_parts)
        where_clause = " AND ".join([predicate.alias_string() for predicate in where_clause_predicates])
        if len(where_clause) > 0:
            where_clause = "\nWHERE " + where_clause
        if forced_select_clause:
            select_clause = forced_select_clause
        else:
            select_clause_strings = []
            for select_clause_element in self._select_clause_elements:
                if select_clause_element.table_occurrence is None:
                    column_string = "*"
                elif select_clause_element.column is None:
                    column_string = "%s.*" % select_clause_element.table_occurrence.alias()
                else:
                    column_string = "%s.%s" % (select_clause_element.table_occurrence.alias(), select_clause_element.column.name())
                if select_clause_element.aggregation_type == SelectClauseAggregationType.NONE:
                    select_clause_strings.append(column_string)
                elif select_clause_element.aggregation_type == SelectClauseAggregationType.MIN:
                    select_clause_strings.append("MIN(%s)" % column_string)
                elif select_clause_element.aggregation_type == SelectClauseAggregationType.MAX:
                    select_clause_strings.append("MAX(%s)" % column_string)
                elif select_clause_element.aggregation_type == SelectClauseAggregationType.COUNT:
                    select_clause_strings.append("COUNT(%s)" % column_string)
                else:
                    raise NotImplementedError()
            select_clause = ", ".join(select_clause_strings)
        return "SELECT %s\nFROM %s%s;" % (select_clause, from_clause, where_clause)


    def _traverse(self) -> List[Tuple[TableOccurrence, List[Join]]]:
        if self._join_graph_edge_dict is None:
            self._join_graph_edge_dict = SPJQuery._get_join_graph_edge_dict(self._table_occurrences, self._joins)
        edge_dict = self._join_graph_edge_dict
        visited = set()
        queue = [list(self._table_occurrences)[0]]
        traversal = []
        while len(queue) > 0:
            current = queue.pop(0)
            if current in visited:
                continue
            current_join_dict = {}
            for other_table_occurrence, join in edge_dict[current]:
                if other_table_occurrence not in visited:
                    queue.append(other_table_occurrence)
                if join not in current_join_dict:
                    new_equivalence_class = []
                    connected = False
                    for ec_table_occurrence, ec_column in join.equivalence_class():
                        if ec_table_occurrence == current:
                            new_equivalence_class.append((ec_table_occurrence, ec_column))
                        elif ec_table_occurrence in visited and not connected:
                            new_equivalence_class.append((ec_table_occurrence, ec_column))
                            connected = True
                    if len(new_equivalence_class) > 1:
                        new_join = Join(new_equivalence_class)
                        current_join_dict[join] = new_join
                    else:
                        current_join_dict[join] = None
            traversal.append((current, list(join for join in current_join_dict.values() if join is not None)))
            visited.add(current)
        assert len(visited) == len(self._table_occurrences)
        return traversal


    def join_graph_edge_dict(self) -> Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]]:
        if self._join_graph_edge_dict is None:
            self._join_graph_edge_dict = SPJQuery._get_join_graph_edge_dict(self._table_occurrences, self._joins)
        return self._join_graph_edge_dict

    @staticmethod
    def _get_join_graph_edge_dict(tabel_occurrences: Iterable[TableOccurrence], joins: Iterable[Join]) -> Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]]:
        edge_dict = {table_occurrence: [] for table_occurrence in tabel_occurrences}
        for join in joins:
            equivalence_class = list(join.equivalence_class())
            for i, (table_occurrence, _) in enumerate(equivalence_class):
                for other_table_occurrence, _ in equivalence_class[i + 1:]:
                    edge_dict[table_occurrence].append((other_table_occurrence, join))
                    edge_dict[other_table_occurrence].append((table_occurrence, join))
        return edge_dict

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
                viable_assignments.extend(SPJQuery._build_viable_assignments(table_assignment_candidates, current_assignment, locked_other_occurrences))
                del current_assignment[chosen_occurrence]
                locked_other_occurrences.remove(other_occurrence)
        return viable_assignments

    def known_equal_assignment(self, other: SPJQuery) -> Optional[Dict[TableOccurrence, TableOccurrence]]:
        own_table_occurrences = self.table_occurrences_by_table()
        other_table_occurrences = other.table_occurrences_by_table()
        if own_table_occurrences.keys() != other_table_occurrences.keys():
            return None
        for table in own_table_occurrences:
            if len(own_table_occurrences[table]) != len(other_table_occurrences[table]):
                return None
        assignment_candidates = {}
        for table in own_table_occurrences:
            table_assignment_candidates = {}
            other_table_hashes = {}
            for other_table_occurrence in other_table_occurrences[table]:
                other_table_hash = other_table_occurrence.predicate().simple_hash()
                if other_table_hash not in other_table_hashes:
                    other_table_hashes[other_table_hash] = []
                other_table_hashes[other_table_hash].append(other_table_occurrence)
            for table_occurrence in own_table_occurrences[table]:
                table_hash = table_occurrence.predicate().simple_hash()
                if table_hash not in other_table_hashes:
                    return None
                assigned_tables = []
                for other_table_occurrence in other_table_hashes[table_hash]:
                    if table_occurrence.known_equal(other_table_occurrence):
                        assigned_tables.append(other_table_occurrence)
                if len(assigned_tables) == 0:
                    return None
                table_assignment_candidates[table_occurrence] = assigned_tables
            viable_assignments = SPJQuery._build_viable_assignments(table_assignment_candidates, {}, set())
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

    def known_equal(self, other: SPJQuery) -> bool:
        return self.known_equal_assignment(other) is not None

    def _assignment_equality(self, other: SPJQuery, table_occurrence_mapping: Dict[TableOccurrence, TableOccurrence]) -> bool:
        if len(self._non_equi_join_predicates) > 0 or len(other.non_equi_join_predicates()) > 0:
            raise NotImplementedError()
        other_join_predicate = other.join_predicate()
        return self._join_predicate.semantically_equal(other_join_predicate, table_occurrence_mapping=table_occurrence_mapping)

    def induced_subquery(self, table_occurrences: List[TableOccurrence]) -> SPJQuery:
        if len(self._non_equi_join_predicates) > 0:
            raise NotImplementedError()
        table_occurrences = set(table_occurrences)
        join_graph_edge_dict = self.join_graph_edge_dict()
        induced_joins = []
        processed_joins = set()
        for table_occurrence in table_occurrences:
            assert table_occurrence in self._table_occurrences
            for _, join in join_graph_edge_dict[table_occurrence]:
                if join not in processed_joins:
                    processed_joins.add(join)
                    equivalence_class = []
                    for eq_table_occurrence, eq_column in join.equivalence_class():
                        if eq_table_occurrence in table_occurrences:
                            equivalence_class.append((eq_table_occurrence, eq_column))
                    if len(equivalence_class) > 1:
                        induced_joins.append(Join(equivalence_class))
        return SPJQuery(list(table_occurrences), induced_joins, [])

    def replace_table_occurrence(self, old_table_occurrence: TableOccurrence, new_table_occurrence: TableOccurrence) -> SPJQuery:
        if len(self._non_equi_join_predicates) > 0:
            raise NotImplementedError()
        table_occurrences = []
        joins = []
        join_graph_edge_dict = self.join_graph_edge_dict()
        processed_joins = set()
        for table_occurrence in self._table_occurrences:
            if table_occurrence == old_table_occurrence:
                table_occurrences.append(new_table_occurrence)
            else:
                table_occurrences.append(table_occurrence)
            for _, join in join_graph_edge_dict[table_occurrence]:
                if join not in processed_joins:
                    processed_joins.add(join)
                    equivalence_class = []
                    for eq_table_occurrence, eq_column in join.equivalence_class():
                        if eq_table_occurrence == old_table_occurrence:
                            equivalence_class.append((new_table_occurrence, eq_column))
                        else:
                            equivalence_class.append((eq_table_occurrence, eq_column))
                    joins.append(Join(equivalence_class))
        return SPJQuery(table_occurrences, joins, [])

    def remove_predicate(self, table_occurrence: TableOccurrence) -> SPJQuery:
        new_table_occurrence = TableOccurrence(table_occurrence.table(), table_occurrence.alias())
        new_table_occurrence.set_predicate(TruePredicate())
        return self.replace_table_occurrence(table_occurrence, new_table_occurrence)

    def is_connected(self) -> bool:
        if len(self._table_occurrences) == 0:
            return True
        edge_dict = self.join_graph_edge_dict()
        visited = set()
        queue = [list(self._table_occurrences)[0]]
        while len(queue) > 0:
            current = queue.pop(0)
            if current in visited:
                continue
            for other_table_occurrence, _ in edge_dict[current]:
                if other_table_occurrence not in visited:
                    queue.append(other_table_occurrence)
            visited.add(current)
        return len(visited) == len(self._table_occurrences)

    def columns_are_unique(self, table_occurrence: TableOccurrence, columns: List[Column]) -> bool:
        # We need to fulfill two conditions to guarantee a column is unique:

        # 1. Base table uniqueness:
        # The provided columns must form a superset of a unique index's columns.
        columns_set = frozenset(columns)
        base_columns_are_unique = False
        for index in table_occurrence.table().indexes():
            if index.unique() and columns_set.issuperset(frozenset(index.columns())):
                base_columns_are_unique = True
                break

        if not base_columns_are_unique:
            return False

        # 2. All other table occurrences must exclusively be "downstream" of this table occurrence via joins to unique columns
        edge_dict = self.join_graph_edge_dict()
        upstream_dict: Dict[TableOccurrence, TableOccurrence] = {}
        queue = [table_occurrence]
        while len(queue) > 0:
            current_table_occurrence = queue.pop(0)
            # Collect all join columns per neighbor across all joins (handles conjunctive join conditions)
            other_to_join_columns: Dict[TableOccurrence, List[Column]] = {}
            for other_table_occurrence, join in edge_dict[current_table_occurrence]:
                if other_table_occurrence in upstream_dict or other_table_occurrence == table_occurrence:
                    continue
                if other_table_occurrence not in other_to_join_columns:
                    other_to_join_columns[other_table_occurrence] = []
                for join_table_occurrence, join_column in join.equivalence_class():
                    if join_table_occurrence == other_table_occurrence:
                        other_to_join_columns[other_table_occurrence].append(join_column)
            for other_table_occurrence, join_columns in other_to_join_columns.items():
                join_columns_set = frozenset(join_columns)
                found_index = False
                for index in other_table_occurrence.table().indexes():
                    if index.unique() and join_columns_set.issuperset(frozenset(index.columns())):
                        upstream_dict[other_table_occurrence] = current_table_occurrence
                        queue.append(other_table_occurrence)
                        found_index = True
                        break
                if not found_index:
                    return False
        return len(upstream_dict) == len(self._table_occurrences) - 1


    def __hash__(self):
        if self._hash is None:
            table_set = frozenset([table_occurrence.simple_hash() for table_occurrence in self._table_occurrences])
            join_set = frozenset([frozenset([(table_occurrence.table(), column) for table_occurrence, column in join.equivalence_class()]) for join in self._joins])
            self._hash = hash((table_set, join_set, len(self._non_equi_join_predicates)))
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SPJQuery):
            return False
        elif self is other:
            return True
        else:
            return self.known_equal(other)



