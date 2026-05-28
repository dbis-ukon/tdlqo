import logging
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from typing import Optional, List, Type, Tuple, Generator, Dict, Set

from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery
from queries.predicates.comparison_operator import COMPARISON_OPERATOR_EQ
from queries.predicates.conjunction import Conjunction
from queries.predicates.join import Join
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression

import concurrent.futures

from schemas.index import Index


class NestedLoopJoinMode(Enum):
    # Allows every NL shape, including non-index inner under a single-table outer (relies on PG actually inserting Materialize — which it won't when it estimates outer=1 row, hence the Rows hint workaround in pg_hint_plan_execution_engine).
    ALL = "all"
    # Allows non-index inner only when the outer is multi-table. Previous default; avoids the missing-Materialize trap for single-table outers but still exposes bushy/right-deep NL blow-ups when cardinality estimates collapse to 1.
    ALL_EXCEPT_NON_INDEX_SINGLE_TABLE = "all_except_non_index_single_table"
    # Inner must be an index scan. The safe default: eliminates the O(outer × inner) catastrophe that occurs when the estimator underestimates either side of a non-index-inner NL.
    INDEX_ONLY = "index_only"


class TopDownOptimizer(Optimizer):
    def __init__(self, type_name: str, name: str, description: str, available_operators: List[Type[RelationalAlgebraExpression]], nested_loop_join_mode: NestedLoopJoinMode = NestedLoopJoinMode.INDEX_ONLY):
        super().__init__(type_name, name, description)
        self.available_operators = sorted(set(available_operators), key=lambda op: op.__name__)
        self.nested_loop_join_mode = nested_loop_join_mode

    def optimize(self, query: Query, explore: bool = False, benchmark_query: Optional[BenchmarkQuery] = None, debug_logger: Optional[logging.Logger] = None) -> Optional[RelationalAlgebraExpression]:
        self._debug_logger = debug_logger
        try:
            self._init_optimization(query)
            group = GroupRelationalAlgebraExpression(query, Requirements())
            return self._optimize(group)
        finally:
            self._debug_logger = None

    def _optimize(self, group: GroupRelationalAlgebraExpression) -> Optional[RelationalAlgebraExpression]:
        memoized_expressions = self.enumerate_memoized_expressions(group)
        chosen_memoized_expression = self._choose(group, memoized_expressions)
        if chosen_memoized_expression is None:
            return None

        child_expressions = []
        for child_group in chosen_memoized_expression.children:
            child_expression = self._optimize(child_group)
            child_expressions.append(child_expression)

        chosen_expression = chosen_memoized_expression.replace_children(child_expressions, True)
        return chosen_expression

    def enumerate_memoized_expressions(self, group: GroupRelationalAlgebraExpression) -> List[RelationalAlgebraExpression]:
        query = group.query()
        assert isinstance(query, SPJQuery)
        table_occurrences = query.table_occurrences()
        memoized_expressions = []
        if len(table_occurrences) == 1:
            table_occurrence = list(table_occurrences)[0]
            if group.requirements.force_index_scan is None:
                if SequentialScanExpression in self.available_operators:
                    memoized_expressions.append(SequentialScanExpression(table_occurrence, query=query))
                if IndexScanExpression in self.available_operators:
                    for index, predicate in self._get_viable_indexes(table_occurrence):
                        memoized_expressions.append(IndexScanExpression(table_occurrence, index, False, False, query=query))
            else:
                if IndexScanExpression in self.available_operators:
                    index, memoized = group.requirements.force_index_scan
                    memoized_expressions.append(IndexScanExpression(table_occurrence, index, memoized, True, query=query))
        elif len(table_occurrences) > 1:
            for first_group, second_group, inter_group_joins in self._enumerate_joins(query):
                memoized_expressions += self._enumerate_join_implementations(query, first_group, second_group, inter_group_joins)
        else:
            raise ValueError("No table occurrences in query")
        return memoized_expressions

    def _get_viable_indexes(self, table_occurrence: TableOccurrence) -> List[Tuple[Index, Predicate]]:
        predicate = table_occurrence.predicate()
        if isinstance(predicate, Conjunction):
            predicates = predicate.predicates()
        else:
            predicates = [predicate]
        filtered_predicates = {}
        for predicate in predicates:
            if isinstance(predicate, SimplePredicate) and predicate.operator() == COMPARISON_OPERATOR_EQ:
                column = predicate.column()
                if column not in filtered_predicates:
                    filtered_predicates[column] = []
                filtered_predicates[column].append(predicate)
        if len(filtered_predicates) == 0:
            return []
        table = table_occurrence.table()
        viable_indexes = []
        for index in table.indexes():
            index_columns = index.columns()
            primary_index_column = index_columns[0]
            index_column_predicates = filtered_predicates.get(primary_index_column, None)
            if index_column_predicates is None:
                continue
            if len(index_column_predicates) > 1:
                index_column_predicate = Conjunction(index_column_predicates)
            else:
                index_column_predicate = index_column_predicates[0]
            viable_indexes.append((index, index_column_predicate))
        return viable_indexes

    @staticmethod
    def _join_graph_connected_acyclic(query: SPJQuery) -> bool:
        # TODO: check if connected
        joins = query.joins()
        node_count = len(query.table_occurrences()) + len(joins)
        edge_count = sum([len(join.equivalence_class()) for join in joins])
        return edge_count == node_count - 1

    @staticmethod
    def _powerset(sequence: List[TableOccurrence]) -> Generator[List[TableOccurrence], None, None]:
        if len(sequence) == 0:
            yield []
        else:
            for item in TopDownOptimizer._powerset(sequence[1:]):
                yield [sequence[0]] + item
                yield item

    @staticmethod
    def _expand_group(group: List[TableOccurrence],
                      edge_dict: Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]],
                      forbidden_joins: Optional[List[Join]] = None) -> Tuple[List[TableOccurrence], List[Join]]:
        if forbidden_joins is None:
            forbidden_joins = set()
        else:
            forbidden_joins = set(forbidden_joins)
        node_queue = group.copy()
        expanded_group = set(group)
        used_joins = set()
        while len(node_queue) > 0:
            current_node = node_queue.pop()
            for neighbor, join in edge_dict[current_node]:
                if join not in forbidden_joins:
                    used_joins.add(join)
                    if neighbor not in expanded_group:
                        expanded_group.add(neighbor)
                        node_queue.append(neighbor)
        return list(expanded_group), list(used_joins)

    @staticmethod
    def _split_join(first_group: Set[TableOccurrence], second_group: Set[TableOccurrence], join: Join) -> Tuple[Optional[Join], Optional[Join]]:
        first_split_join_equivalence_class = []
        second_split_join_equivalence_class = []
        for table_occurrence, column in join.equivalence_class():
            if table_occurrence in first_group:
                first_split_join_equivalence_class.append((table_occurrence, column))
            elif table_occurrence in second_group:
                second_split_join_equivalence_class.append((table_occurrence, column))
            else:
                raise ValueError("Table occurrence not in either group")
        if len(first_split_join_equivalence_class) > 1:
            first_split_join = Join(first_split_join_equivalence_class, redundant=join.is_redundant())
        else:
            first_split_join = None
        if len(second_split_join_equivalence_class) > 1:
            second_split_join = Join(second_split_join_equivalence_class, redundant=join.is_redundant())
        else:
            second_split_join = None
        return first_split_join, second_split_join

    @staticmethod
    def _split_acyclic_query(split_join: Join,
                             edge_dict: Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]],
                             first_group: List[TableOccurrence],
                             second_group: List[TableOccurrence]) -> Tuple[Query, Query, Join]:
        first_group_expanded, first_group_used_joins = TopDownOptimizer._expand_group(first_group, edge_dict, forbidden_joins=[split_join])
        second_group_expanded, second_group_used_joins = TopDownOptimizer._expand_group(second_group, edge_dict, forbidden_joins=[split_join])

        first_split_join, second_split_join = TopDownOptimizer._split_join(set(first_group_expanded), set(second_group_expanded), split_join)
        if first_split_join is not None:
            first_group_used_joins.append(first_split_join)
        if second_split_join is not None:
            second_group_used_joins.append(second_split_join)

        first_group_query = SPJQuery(first_group_expanded, first_group_used_joins, [])
        second_group_query = SPJQuery(second_group_expanded, second_group_used_joins, [])
        return first_group_query, second_group_query, split_join

    @staticmethod
    def _enumerate_joins(query: SPJQuery) -> List[Tuple[Query, Query, List[Join]]]:
        if TopDownOptimizer._join_graph_connected_acyclic(query):
            return TopDownOptimizer._enumerate_joins_acyclic_graph(query)
        else:
            return TopDownOptimizer._enumerate_joins_brute_force(query)

    @staticmethod
    def _enumerate_joins_acyclic_graph(query: SPJQuery) -> List[Tuple[Query, Query, List[Join]]]:
        join_graph_edge_dict = query.join_graph_edge_dict()
        enumerated_joins = []
        for join in query.joins():
            table_occurrences = sorted(set(table_occurrence for table_occurrence, _ in join.equivalence_class()), key=lambda to: to.sort_key())
            assert len(table_occurrences) > 1
            first_table_occurrence = table_occurrences[0]
            other_table_occurrences = table_occurrences[1:]
            for subset in TopDownOptimizer._powerset(other_table_occurrences):
                first_group = [first_table_occurrence] + subset  # Forcing the first table occurrence to be in the first group is a convenient way to avoid duplicate joins
                first_group_set = set(first_group)
                second_group = [table_occurrence for table_occurrence in other_table_occurrences if table_occurrence not in first_group_set]
                if len(second_group) > 0:
                    first_group, second_group, inter_group_join = TopDownOptimizer._split_acyclic_query(join, join_graph_edge_dict, first_group, second_group)
                    enumerated_joins.append((first_group, second_group, [inter_group_join]))
        return enumerated_joins

    @staticmethod
    def _enumerate_joins_brute_force(query: SPJQuery) -> List[Tuple[Query, Query, List[Join]]]:
        join_graph_edge_dict = query.join_graph_edge_dict()
        enumerated_joins = []
        table_occurrences = query.table_occurrences()
        assert len(table_occurrences) > 1
        table_occurrences_list = sorted(table_occurrences, key=lambda to: to.sort_key())
        first_table_occurrence = table_occurrences_list[0]
        other_table_occurrences = table_occurrences_list[1:]
        for subset in TopDownOptimizer._powerset(other_table_occurrences):
            first_group = [first_table_occurrence] + subset
            first_group_set = set(first_group)
            second_group = [table_occurrence for table_occurrence in other_table_occurrences if table_occurrence not in first_group_set]
            if len(second_group) > 0 and TopDownOptimizer._is_group_connected(first_group, join_graph_edge_dict) and TopDownOptimizer._is_group_connected(second_group, join_graph_edge_dict):
                first_group, second_group, inter_group_joins = TopDownOptimizer._split_query(first_group, second_group, query)
                enumerated_joins.append((first_group, second_group, inter_group_joins))
        return enumerated_joins

    @staticmethod
    def _is_group_connected(group: List[TableOccurrence], edge_dict: Dict[TableOccurrence, List[Tuple[TableOccurrence, Join]]]) -> bool:
        if len(group) == 1:
            return True
        node_set = set(group)
        queue = [group[0]]
        visited_nodes = set(queue)
        while len(queue) > 0:
            current_node = queue.pop()
            for neighbor, _ in edge_dict[current_node]:
                if neighbor not in visited_nodes and neighbor in node_set:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    if len(visited_nodes) == len(node_set):
                        return True
        return False

    @staticmethod
    def _split_query(first_group: List[TableOccurrence],
                     second_group: List[TableOccurrence],
                     query: SPJQuery) -> Tuple[Query, Query, List[Join]]:
        first_group_set = set(first_group)
        second_group_set = set(second_group)
        first_group_joins = []
        second_group_joins = []
        mixed_joins = []
        for join in query.joins():
            has_first_group = False
            has_second_group = False
            for table_occurrence, _ in join.equivalence_class():
                if table_occurrence in first_group_set:
                    has_first_group = True
                    if has_second_group:
                        break
                elif table_occurrence in second_group_set:
                    has_second_group = True
                    if has_first_group:
                        break
                else:
                    raise ValueError("Table occurrence not in either group")
            if has_first_group and not has_second_group:
                first_group_joins.append(join)
            elif has_second_group and not has_first_group:
                second_group_joins.append(join)
            else:
                mixed_joins.append(join)

        inter_group_joins = []
        for join in mixed_joins:
            first_group_join, second_group_join = TopDownOptimizer._split_join(first_group_set, second_group_set, join)
            if first_group_join is not None:
                first_group_joins.append(first_group_join)
            if second_group_join is not None:
                second_group_joins.append(second_group_join)
            inter_group_joins.append(join)

        first_group_query = SPJQuery(first_group, first_group_joins, [])
        second_group_query = SPJQuery(second_group, second_group_joins, [])
        return first_group_query, second_group_query, inter_group_joins

    def _enumerate_join_implementations(self,
                                        query: SPJQuery,
                                        first_group: Query,
                                        second_group: Query,
                                        joins: List[Join]) -> List[RelationalAlgebraExpression]:
        join_implementations = []
        if len(joins) == 0:
            join_condition = TruePredicate()
        elif len(joins) == 1:
            join_condition = joins[0]
        else:
            join_condition = Conjunction(joins)
        only_redundant_joins = all(join.is_redundant() for join in joins)
        for available_operation in self.available_operators:
            if only_redundant_joins and not issubclass(available_operation, NestedLoopJoinExpression):
                continue
            if issubclass(available_operation, NestedLoopJoinExpression):
                def _process_nested_loop_implementation(outer_group: SPJQuery, inner_group: SPJQuery) -> List[RelationalAlgebraExpression]:
                    implementations = []
                    num_inner_table_occurrences = len(inner_group.table_occurrences())
                    outer_group_expression = GroupRelationalAlgebraExpression(outer_group, Requirements())
                    inner_group_expressions = []
                    if num_inner_table_occurrences == 1:
                        if IndexScanExpression in self.available_operators:
                            inner_table_occurrence = list(inner_group.table_occurrences())[0]
                            inner_join_columns = set([column for join in joins for table_occurrence, column in join.equivalence_class() if table_occurrence == inner_table_occurrence])
                            # Collect equality predicate columns from the inner table.
                            inner_predicate = inner_table_occurrence.predicate()
                            equality_predicate_columns = set()
                            if isinstance(inner_predicate, Conjunction):
                                for p in inner_predicate.predicates():
                                    if isinstance(p, SimplePredicate) and p.operator() == COMPARISON_OPERATOR_EQ:
                                        equality_predicate_columns.add(p.column())
                            elif isinstance(inner_predicate, SimplePredicate) and inner_predicate.operator() == COMPARISON_OPERATOR_EQ:
                                equality_predicate_columns.add(inner_predicate.column())
                            useful_columns = inner_join_columns | equality_predicate_columns
                            # Pick the best index: longest contiguous useful prefix, then unique.
                            best_index = None
                            best_score = None
                            for index in inner_table_occurrence.table().indexes():
                                if index.columns()[0] not in useful_columns:
                                    continue
                                prefix_has_join_column = False
                                useful_prefix_length = 0
                                for col in index.columns():
                                    if col in useful_columns:
                                        useful_prefix_length += 1
                                        if col in inner_join_columns:
                                            prefix_has_join_column = True
                                    else:
                                        break
                                if not prefix_has_join_column:
                                    continue
                                score = (index.prefix_distinct_count(useful_prefix_length), -index.level(), index.name())
                                if best_index is None or score > best_score:
                                    best_score = score
                                    best_index = index
                            if best_index is not None:
                                inner_group_expressions.append(GroupRelationalAlgebraExpression(inner_group, Requirements(force_index_scan=(best_index, True))))
                                inner_group_expressions.append(GroupRelationalAlgebraExpression(inner_group, Requirements(force_index_scan=(best_index, False))))
                    mode = self.nested_loop_join_mode
                    allow_non_index_inner = mode == NestedLoopJoinMode.ALL or (mode == NestedLoopJoinMode.ALL_EXCEPT_NON_INDEX_SINGLE_TABLE and len(outer_group.table_occurrences()) > 1)
                    if allow_non_index_inner:
                        inner_group_expressions.append(GroupRelationalAlgebraExpression(inner_group, Requirements()))
                    for inner_group_expression in inner_group_expressions:
                        implementations.append(available_operation.build_expression(outer_group_expression, inner_group_expression, join_condition, query=query))
                    return implementations
                assert isinstance(first_group, SPJQuery)
                assert isinstance(second_group, SPJQuery)
                join_implementations += _process_nested_loop_implementation(first_group, second_group)
                join_implementations += _process_nested_loop_implementation(second_group, first_group)
            elif issubclass(available_operation, JoinExpression):
                first_group_expression = GroupRelationalAlgebraExpression(first_group, Requirements())
                second_group_expression = GroupRelationalAlgebraExpression(second_group, Requirements())
                join_implementations.append(available_operation.build_expression(first_group_expression, second_group_expression, join_condition, query=query))
                join_implementations.append(available_operation.build_expression(second_group_expression, first_group_expression, join_condition, query=query))
        return join_implementations

    @abstractmethod
    def _init_optimization(self, query: Query):
        pass

    @abstractmethod
    def _choose(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> RelationalAlgebraExpression:
        pass


    @staticmethod
    def enumerate_logical_groups(query: SPJQuery) -> List[SPJQuery]:
        query = SPJQuery(query.table_occurrences(), query.joins(), query.non_equi_join_predicates(), None)
        group_queue = [query]
        logical_group_identifiers = set()
        logical_groups = []
        while len(group_queue) > 0:
            current_group = group_queue.pop()
            current_group_table_occurrences = current_group.table_occurrences()
            if current_group_table_occurrences in logical_group_identifiers:
                continue
            logical_group_identifiers.add(current_group_table_occurrences)
            logical_groups.append(current_group)
            if len(current_group_table_occurrences) == 1:
                continue
            for first_group, second_group, _ in TopDownOptimizer._enumerate_joins(current_group):
                group_queue.append(first_group)
                group_queue.append(second_group)
        return logical_groups








