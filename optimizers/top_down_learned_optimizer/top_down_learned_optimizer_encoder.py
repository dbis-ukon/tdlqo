import json
import math
import os
from typing import List, TypeVar, Dict, Iterable, Tuple, Set, Optional, FrozenSet

import numpy as np
import torch

from cost_models.postgresql_cost_model import PostgreSQLCostModel
from optimizers.top_down_learned_optimizer.memory.memoized_expression_memory import MemoizedExpressionMemory
from optimizers.top_down_learned_optimizer.memory.group_memory import GroupMemory
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_training_data import \
    TopDownLearnedOptimizerTrainingData
from queries.predicates.comparison_operator import COMPARISON_OPERATOR_EQ
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from sample_encoders.sample_encoder import SampleEncoder
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import TopDownLearnedOptimizerConfiguration
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import TopDownLearnedOptimizerData
from queries.predicates.conjunction import Conjunction
from queries.predicates.disjunction import Disjunction
from queries.predicates.join import Join
from queries.predicates.negation import Negation
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.predicates.true_predicate import TruePredicate
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from schemas.column import Column
from schemas.schema import Schema
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import TopDownLearnedOptimizerTargetData, PlanType


class TopDownLearnedOptimizerEncoder:
    def __init__(self, configuration: TopDownLearnedOptimizerConfiguration, schema: Schema):
        tables = list(schema.tables())
        tables.sort(key=lambda t: t.name())
        self._table_encodings, self._table_index = self._build_one_hot_encodings(tables)
        self._column_orders = {}
        for table in schema.tables():
            table_columns = list(table.columns())
            self._column_orders[table] = table_columns
        key_columns = set()
        for foreign_key in schema.foreign_keys():
            for column in foreign_key.mapping().keys():
                key_columns.add((foreign_key.foreign_key_table(), column))
            for column in foreign_key.mapping().values():
                key_columns.add((foreign_key.primary_key_table(), column))
        key_columns = list(key_columns)
        key_columns.sort(key=lambda x: (x[0].name(), x[1].name()))
        key_columns = [column for table, column in key_columns]
        # Non-foreign-key join columns are registered on demand in _key_column_index, growing the one-hot encoding.
        self._key_column_indexes = {column: i for i, column in enumerate(key_columns)}
        self._last_num_key_columns = len(self._key_column_indexes)
        self._table_occurrence_encoding_cardinality_estimator = configuration.table_occurrence_encoding_cardinality_estimator
        self._table_occurrence_encoding_predicate_columns = configuration.table_occurrence_encoding_predicate_columns
        self._table_occurrence_encoding_predicate_costs = configuration.table_occurrence_encoding_predicate_costs and isinstance(configuration.cost_model, PostgreSQLCostModel)
        self._table_occurrence_encoding_index_viable = configuration.table_occurrence_encoding_index_viable and IndexScanExpression in configuration.available_operators
        self._join_operator_table_occurrence_encoding_neighborhood = configuration.join_operator_table_occurrence_encoding_neighborhood
        self._cost_model = configuration.cost_model
        if configuration.table_occurrence_encoding_random_sample:
            self._random_sample_encoder = SampleEncoder.random_sample_encoder(schema, configuration.table_occurrence_encoding_random_sample)
        else:
            self._random_sample_encoder = None
        if configuration.table_occurrence_encoding_feature_selected_sample:
            self._feature_selected_sample_encoder = SampleEncoder.empty_sample_encoder(schema, configuration.table_occurrence_encoding_feature_selected_sample)
        else:
            self._feature_selected_sample_encoder = None
        current_column_index = 0
        self._column_index_dict = {}
        for table in sorted(schema.tables(), key=lambda t: t.name()):
            for column in sorted(table.columns(), key=lambda c: c.name()):
                self._column_index_dict[column] = current_column_index
                current_column_index += 1
        join_operators = set()
        scan_operators = set()
        requirements = set()
        requirements.add(False)
        self._distinguish_nested_loops = False
        for operator in configuration.available_operators:
            if issubclass(operator, JoinExpression):
                if issubclass(operator, NestedLoopJoinExpression) and IndexScanExpression in configuration.available_operators:
                    join_operators.add((operator, True, True))
                    requirements.add((True, True))
                    join_operators.add((operator, True, False))
                    requirements.add((True, False))
                    join_operators.add((operator, False, False))
                    self._distinguish_nested_loops = True
                else:
                    join_operators.add(operator)
            elif issubclass(operator, ScanExpression):
                scan_operators.add(operator)
            else:
                raise ValueError("Unknown operator type")
        # Sorted by their string form: the elements are operator classes, (class, bool, bool) tuples
        # and requirement values, which have no common ordering but a stable textual one.
        self._join_operator_encodings, _ = self._build_one_hot_encodings(sorted(join_operators, key=str))
        self._scan_operator_encodings, _ = self._build_one_hot_encodings(sorted(scan_operators, key=str))
        self._requirement_encodings, _ = self._build_one_hot_encodings(sorted(requirements, key=str))
        self._learn_decomposed_costs = configuration.learn_decomposed_costs
        self._recursive_cost_constraint = configuration.recursive_cost_constraint_factor is not None
        self._last_training_queries = set()

        self.version = 1

    def num_tables(self) -> int:
        return len(self._table_encodings)

    def table_occurrence_information_size(self) -> int:
        size = self.num_tables()
        if self._table_occurrence_encoding_cardinality_estimator is not None:
            size += 1
        if self._table_occurrence_encoding_predicate_columns:
            size += len(self._column_index_dict)
        if self._table_occurrence_encoding_predicate_costs:
            size += 1
        if self._table_occurrence_encoding_index_viable:
            size += 1
        return size

    def table_specific_information_size(self) -> int:
        size = 0
        if self._random_sample_encoder is not None:
            size += self._random_sample_encoder.max_sample_size()
        if self._feature_selected_sample_encoder is not None:
            size += self._feature_selected_sample_encoder.max_sample_size()
        return size

    def join_information_size(self) -> int:
        return len(self._key_column_indexes) * 2

    def _key_column_index(self, column: Column) -> int:
        index = self._key_column_indexes.get(column)
        if index is None:
            index = len(self._key_column_indexes)
            self._key_column_indexes[column] = index
        return index

    def _key_column_encoding(self, column: Column, cache: Dict[Column, torch.FloatTensor]) -> torch.FloatTensor:
        encoding = cache.get(column)
        if encoding is None:
            encoding = torch.zeros(len(self._key_column_indexes))
            encoding[self._key_column_indexes[column]] = 1
            cache[column] = encoding
        return encoding

    def join_operator_information_size(self) -> int:
        return len(self._join_operator_encodings)

    def scan_operator_information_size(self) -> int:
        return len(self._scan_operator_encodings)

    def requirement_information_size(self) -> int:
        return len(self._requirement_encodings)

    def join_operator_subgroup_table_information_size(self) -> int:
        size = 0
        if self._join_operator_table_occurrence_encoding_neighborhood:
            size += self.num_tables()
        return size

    T = TypeVar('T')
    @staticmethod
    def _build_one_hot_encodings(objects: Iterable[T]) -> Tuple[Dict[T, torch.FloatTensor], Dict[T, int]]:
        # dict.fromkeys removes duplicates while keeping the caller's order. A set would order by
        # identity hash instead, which differs between processes, so a module saved in one process
        # would read its table and operator one-hots in a different order in the next one.
        objects = list(dict.fromkeys(objects))
        encoding_dict = {}
        index_dict = {}
        for i, obj in enumerate(objects):
            encoding = torch.zeros(len(objects))
            encoding[i] = 1
            encoding_dict[obj] = encoding
            index_dict[obj] = i
        return encoding_dict, index_dict

    def encode(self,
               group: GroupRelationalAlgebraExpression,
               possible_plans: List[RelationalAlgebraExpression],
               table_occurrence_encodings: Optional[Dict[TableOccurrence, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None) -> Tuple[TopDownLearnedOptimizerData, Dict[Tuple[PlanType, int], RelationalAlgebraExpression], List[GroupRelationalAlgebraExpression]]:
        query = group.query()
        assert isinstance(query, SPJQuery)
        table_occurrence_index_dict = {}
        table_occurrence_information = []
        table_index = []
        table_specific_information = []
        all_table_occurrences = query.table_occurrences()
        for i, table_occurrence in enumerate(all_table_occurrences):
            table_index.append(self._table_index[table_occurrence.table()])
            table_occurrence_index_dict[table_occurrence] = i
            if table_occurrence_encodings is not None and table_occurrence in table_occurrence_encodings:
                table_occurrence_info, table_specific_info = table_occurrence_encodings[table_occurrence]
            else:
                table_occurrence_info, table_specific_info = self._encode_table_occurrence(table_occurrence)
            table_occurrence_information.append(table_occurrence_info)
            table_specific_information.append(table_specific_info)

        join_pair_dict = {i: {} for i in range(len(all_table_occurrences))}
        join_dict = {table_occurrence: set() for table_occurrence in all_table_occurrences}
        # Register all join columns first so that all encodings in this call share the current size.
        for join in query.joins():
            for _, column in join.equivalence_class():
                self._key_column_index(column)
        key_column_encodings = {}
        for join in query.joins():
            equivalence_class = list(join.equivalence_class())
            for i, (first_table_occurrence, first_column) in enumerate(equivalence_class):
                first_column_encoding = self._key_column_encoding(first_column, key_column_encodings)
                first_table_occurrence_index = table_occurrence_index_dict[first_table_occurrence]
                for j, (second_table_occurrence, second_column) in enumerate(equivalence_class):
                    if i == j:
                        continue
                    second_column_encoding = self._key_column_encoding(second_column, key_column_encodings)
                    second_table_occurrence_index = table_occurrence_index_dict[second_table_occurrence]

                    join_pair = (first_column, first_column_encoding,
                                 second_column, second_column_encoding)
                    if first_table_occurrence_index not in join_pair_dict[second_table_occurrence_index]:
                        join_pair_dict[second_table_occurrence_index][first_table_occurrence_index] = (first_table_occurrence, second_table_occurrence, [])
                    join_pair_dict[second_table_occurrence_index][first_table_occurrence_index][2].append(join_pair)
                    join_dict[first_table_occurrence].add(second_table_occurrence)

        join_information = []
        join_information_index = []
        join_index = []
        join_indptrs = []
        join_counts = []
        join_index_dict = {}
        join_information_index_counter = 0
        for second_table_occurrence_index, edge_dict in join_pair_dict.items():
            join_indptrs.append(len(join_index))
            join_counts.append(len(edge_dict))
            for first_table_occurrence_index, (first_table_occurrence, second_table_occurrence, join_pairs) in edge_dict.items():
                for (first_column, first_column_encoding, second_column, second_column_encoding) in join_pairs:
                    join_information.append(torch.cat([first_column_encoding, second_column_encoding]))
                    join_information_index.append(join_information_index_counter)
                join_index_dict[(first_table_occurrence, second_table_occurrence)] = len(join_index)
                join_index.append([first_table_occurrence_index, second_table_occurrence_index])
                join_information_index_counter += 1

        join_operator_information = []
        join_operator_subgroup_table_index = []
        join_operator_subgroup_table_information = []
        join_operator_subgroup_index = []
        join_operator_subgroup_indptrs = []
        join_operator_subgroup_counts = []
        requirement_information = []
        join_operator_outer_subgroup_index = []
        join_operator_inner_subgroup_index = []
        subgroup_dict: Dict[GroupRelationalAlgebraExpression, int] = {}
        subgroups: List[GroupRelationalAlgebraExpression] = []
        join_operator_index = []
        join_operator_join_index = []
        join_operator_join_indptrs = []
        join_operator_join_index_counts = []
        join_operator_num_tables = []
        num_table_occurrences = len(query.table_occurrences())
        scan_operator_information = []
        scan_operator_index = []
        plan_dict = {}
        scan_count = 0
        join_count = 0
        for plan in possible_plans:
            plan_type = type(plan)
            if isinstance(plan, ScanExpression):
                if plan_type not in self._scan_operator_encodings:
                    raise ValueError("Unknown scan operator type")
                scan_operator_information.append(self._scan_operator_encodings[plan_type])
                scan_operator_index.append(0)
                plan_dict[(PlanType.SCAN, scan_count)] = plan
                scan_count += 1
            elif isinstance(plan, JoinExpression):
                if self._distinguish_nested_loops and isinstance(plan, NestedLoopJoinExpression):
                    if isinstance(plan.inner, IndexScanExpression):
                        is_index_scan = True
                        is_memoized = plan.inner.memoized
                    elif isinstance(plan.inner, GroupRelationalAlgebraExpression) and plan.inner.requirements.force_index_scan is not None:
                        is_index_scan = True
                        is_memoized = plan.inner.requirements.force_index_scan[1]
                    else:
                        is_index_scan = False
                        is_memoized = False
                    plan_type = (NestedLoopJoinExpression, is_index_scan, is_memoized)
                if plan_type not in self._join_operator_encodings:
                    raise ValueError("Unknown join operator type")
                join_number = len(join_operator_information)
                join_operator_information.append(self._join_operator_encodings[plan_type])
                join_operator_num_tables.append(num_table_occurrences)
                plan_dict[(PlanType.JOIN, join_count)] = plan
                join_count += 1

                def process_subgroup(subgroup: GroupRelationalAlgebraExpression) -> Tuple[int, FrozenSet[TableOccurrence]]:
                    query = subgroup.query()
                    assert isinstance(query, SPJQuery)
                    table_occurrences = query.table_occurrences()
                    if subgroup not in subgroup_dict:
                        subgroup_number = len(subgroup_dict)
                        subgroup_dict[subgroup] = subgroup_number
                        subgroups.append(subgroup)
                        join_operator_subgroup_indptrs.append(len(join_operator_subgroup_index))
                        for table_occurrence in table_occurrences:
                            table_occurrence_index = table_occurrence_index_dict[table_occurrence]
                            join_operator_subgroup_table_index.append(table_occurrence_index)
                            join_operator_subgroup_index.append(subgroup_number)
                        join_operator_subgroup_counts.append(len(table_occurrences))
                        if subgroup.requirements.has_no_requirement():
                            requirement = False
                        elif subgroup.requirements.force_index_scan is not None:
                            requirement = (True, subgroup.requirements.force_index_scan[1])
                        else:
                            raise ValueError("Unknown requirement type")
                        requirement_information.append(self._requirement_encodings[requirement])
                        if self._join_operator_table_occurrence_encoding_neighborhood:
                            info = torch.zeros(len(table_occurrences), self.num_tables())
                            for i, table_occurrence in enumerate(table_occurrences):
                                for neighbor in join_dict[table_occurrence]:
                                    if neighbor in table_occurrences:
                                        neighbor_table_index = self._table_index[neighbor.table()]
                                        info[i, neighbor_table_index] += 1
                            join_operator_subgroup_table_information.append(info)
                        else:
                            join_operator_subgroup_table_information.append(torch.zeros(len(table_occurrences), 0))
                    return subgroup_dict[subgroup], table_occurrences

                outer_subgroup_number, outer_table_occurrences = process_subgroup(plan.outer)
                inner_subgroup_number, inner_table_occurrences = process_subgroup(plan.inner)
                join_operator_outer_subgroup_index.append(outer_subgroup_number)
                join_operator_inner_subgroup_index.append(inner_subgroup_number)
                join_condition = plan.join_condition
                if isinstance(join_condition, TruePredicate):
                    join_conditions = []
                elif isinstance(join_condition, Conjunction):
                    join_conditions = join_condition.predicates()
                    for jc in join_conditions:
                        assert isinstance(jc, Join)
                elif isinstance(join_condition, Join):
                    join_conditions = [join_condition]
                else:
                    raise ValueError("Unknown join condition type")
                join_operator_join_indptrs.append(len(join_operator_join_index))
                join_operator_join_count = 0
                for join_condition in join_conditions:
                    equivalence_class = list(join_condition.equivalence_class())
                    for i, (first_table_occurrence, first_column) in enumerate(equivalence_class):
                        for j, (second_table_occurrence, second_column) in enumerate(equivalence_class):
                            if i != j and first_table_occurrence in outer_table_occurrences and second_table_occurrence in inner_table_occurrences:
                                join_operator_index.append(join_number)
                                join_operator_join_index.append(join_index_dict[(first_table_occurrence, second_table_occurrence)])
                                join_operator_join_count += 1
                join_operator_join_index_counts.append(join_operator_join_count)
            else:
                raise ValueError("Unknown operator type")

        table_occurrence_information = torch.stack(table_occurrence_information)
        table_index = torch.LongTensor(table_index)
        if len(table_specific_information) == 0:
            table_specific_information = torch.zeros((0, self.table_specific_information_size()))
        else:
            table_specific_information = torch.stack(table_specific_information)
        if len(join_index) == 0:
            join_index = torch.zeros((2, 0), dtype=torch.long)
            join_information = torch.zeros((0, self.join_information_size()))
            join_information_index = torch.zeros((0,), dtype=torch.long)
            join_operator_num_tables = torch.ones((0,))
            join_operator_information = torch.zeros((0, self.join_operator_information_size()))
            join_operator_subgroup_indptrs = torch.zeros((0,), dtype=torch.long)
            join_operator_subgroup_counts = torch.zeros((0, 1), dtype=torch.float32)
            join_operator_join_indptrs = torch.zeros((0,), dtype=torch.long)
            join_operator_join_index_counts = torch.zeros((0, 1), dtype=torch.float32)
            requirement_information = torch.zeros((0, self.requirement_information_size()))
            join_operator_subgroup_table_information = torch.zeros((0, self.num_tables()))
        else:
            join_index = torch.transpose(torch.LongTensor(join_index), 0, 1)
            join_information = torch.stack(join_information)
            join_information_index = torch.LongTensor(join_information_index)
            join_operator_num_tables = torch.FloatTensor(join_operator_num_tables)
            join_operator_information = torch.stack(join_operator_information)
            join_operator_subgroup_indptrs = torch.LongTensor(join_operator_subgroup_indptrs)
            join_operator_subgroup_counts = torch.FloatTensor(join_operator_subgroup_counts).unsqueeze(1)
            join_operator_join_indptrs = torch.LongTensor(join_operator_join_indptrs)
            join_operator_join_index_counts = torch.FloatTensor(join_operator_join_index_counts).unsqueeze(1)
            requirement_information = torch.stack(requirement_information)
            join_operator_subgroup_table_information = torch.cat(join_operator_subgroup_table_information, dim=0)

        join_indptrs = torch.LongTensor(join_indptrs)
        join_counts = torch.FloatTensor(join_counts).unsqueeze(1)

        join_operator_batch = torch.zeros(len(join_operator_information), dtype=torch.long)

        join_operator_subgroup_table_index = torch.LongTensor(join_operator_subgroup_table_index)
        join_operator_subgroup_index = torch.LongTensor(join_operator_subgroup_index)
        join_operator_outer_subgroup_index = torch.LongTensor(join_operator_outer_subgroup_index)
        join_operator_inner_subgroup_index = torch.LongTensor(join_operator_inner_subgroup_index)
        join_operator_index = torch.LongTensor(join_operator_index)
        join_operator_join_index = torch.LongTensor(join_operator_join_index)
        scan_operator_batch = torch.zeros(len(scan_operator_information), dtype=torch.long)
        if len(scan_operator_information) == 0:
            scan_operator_information = torch.zeros((0, self.scan_operator_information_size()))
        else:
            scan_operator_information = torch.stack(scan_operator_information)
        scan_operator_index = torch.LongTensor(scan_operator_index)
        batch_indptr = torch.zeros(1, dtype=torch.long)
        batch_count = torch.FloatTensor([len(all_table_occurrences)]).unsqueeze(1)

        data = TopDownLearnedOptimizerData(table_occurrence_information,
                                           table_index,
                                           table_specific_information,
                                           join_information,
                                           join_information_index,
                                           join_index,
                                           join_indptrs,
                                           join_counts,
                                           join_operator_batch,
                                           join_operator_information,
                                           join_operator_subgroup_table_index,
                                           join_operator_subgroup_table_information,
                                           join_operator_subgroup_index,
                                           join_operator_subgroup_indptrs,
                                           join_operator_subgroup_counts,
                                           requirement_information,
                                           join_operator_outer_subgroup_index,
                                           join_operator_inner_subgroup_index,
                                           join_operator_index,
                                           join_operator_join_index,
                                           join_operator_join_indptrs,
                                           join_operator_join_index_counts,
                                           join_operator_num_tables,
                                           scan_operator_batch,
                                           scan_operator_information,
                                           scan_operator_index,
                                           batch_indptr,
                                           batch_count)
        return data, plan_dict, subgroups

    def _encode_table_occurrence(self, table_occurrence: TableOccurrence) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        table_occurrence_info = [self._table_encodings[table_occurrence.table()]]
        if self._table_occurrence_encoding_cardinality_estimator is not None:
            if isinstance(table_occurrence.predicate(), TruePredicate):
                cardinality = table_occurrence.table().cardinality()
            else:
                cardinality = self._table_occurrence_encoding_cardinality_estimator.estimate(SPJQuery([table_occurrence], [], []))
                assert cardinality is not None
            cardinality = max(cardinality, 1)
            cardinality_encoding = torch.tensor([math.log(cardinality)])
            table_occurrence_info.append(cardinality_encoding)
        if self._table_occurrence_encoding_predicate_columns:
            predicate_column_encoding = torch.zeros(len(self._column_index_dict), dtype=torch.float32)
            self._encode_predicate_columns(table_occurrence.predicate(), predicate_column_encoding)
            table_occurrence_info.append(predicate_column_encoding)
        if self._table_occurrence_encoding_predicate_costs:
            predicate_cost = self._cost_model.cpu_operator_cost_per_tuple(table_occurrence)
            predicate_cost_encoding = torch.tensor([predicate_cost])
            table_occurrence_info.append(predicate_cost_encoding)
        if self._table_occurrence_encoding_index_viable:
            index_viable = self._index_viable(table_occurrence)
            index_viable_encoding = torch.tensor([1.0 if index_viable else 0.0])
            table_occurrence_info.append(index_viable_encoding)
        table_occurrence_info = torch.cat(table_occurrence_info)
        table_specific_info = []
        if self._random_sample_encoder is not None:
            sample_encoding = self._random_sample_encoder.encode(table_occurrence)
            table_specific_info.append(sample_encoding)
        if self._feature_selected_sample_encoder is not None:
            sample_encoding = self._feature_selected_sample_encoder.encode(table_occurrence)
            table_specific_info.append(sample_encoding)
        table_specific_information = torch.cat(table_specific_info)
        return table_occurrence_info, table_specific_information

    def _index_viable(self, table_occurrence: TableOccurrence) -> bool:
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
            return False
        table = table_occurrence.table()
        for index in table.indexes():
            index_columns = index.columns()
            primary_index_column = index_columns[0]
            if primary_index_column in filtered_predicates:
                return True
        return False

    def encode_table_occurrences(self, query: SPJQuery) -> Dict[TableOccurrence, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        table_occurrence_encodings = {}
        for table_occurrence in query.table_occurrences():
            table_occurrence_encodings[table_occurrence] = self._encode_table_occurrence(table_occurrence)
        return table_occurrence_encodings

    def add_table_occurrence_encodings(self, query: SPJQuery, table_occurrence_encodings: Dict[TableOccurrence, Tuple[torch.FloatTensor, torch.FloatTensor]]):
        for table_occurrence in query.table_occurrences():
            if table_occurrence not in table_occurrence_encodings:
                table_occurrence_encodings[table_occurrence] = self._encode_table_occurrence(table_occurrence)

    def _encode_predicate_columns(self, predicate: Predicate, encoding: torch.FloatTensor):
        if isinstance(predicate, TruePredicate):
            return
        elif isinstance(predicate, Conjunction):
            for clause_predicate in predicate.predicates():
                self._encode_predicate_columns(clause_predicate, encoding)
        elif isinstance(predicate, Disjunction):
            for clause_predicate in predicate.predicates():
                self._encode_predicate_columns(clause_predicate, encoding)
        elif isinstance(predicate, Negation):
            self._encode_predicate_columns(predicate.predicate(), encoding)
        elif isinstance(predicate, SimplePredicate):
            column_index = self._column_index_dict[predicate.column()]
            encoding[column_index] = 1
        else:
            raise NotImplementedError

    def encode_pre_training(self,
                            group: GroupRelationalAlgebraExpression,
                            local_costs: List[Tuple[RelationalAlgebraExpression, Optional[float]]],
                            subgroup_min_total_costs: Dict[GroupRelationalAlgebraExpression, Optional[float]]) -> Optional[TopDownLearnedOptimizerTrainingData]:
        valid_costs = [(plan, cost) for plan, cost in local_costs if cost is not None]
        if len(valid_costs) == 0:
            return None
        query_data, plan_dict, subgroups = self.encode(group, [plan for plan, cost in valid_costs])

        plan_type_counts = {plan_type: 0 for plan_type in PlanType}
        for plan_type, _ in plan_dict:
            plan_type_counts[plan_type] += 1

        cost_dict = {}
        for plan, cost in valid_costs:
            cost_dict[plan] = cost

        join_operator_cost_mask = torch.ones(plan_type_counts[PlanType.JOIN], dtype=torch.bool)
        scan_operator_cost_mask = torch.ones(plan_type_counts[PlanType.SCAN], dtype=torch.bool)

        join_operator_min_costs = np.ones(plan_type_counts[PlanType.JOIN])
        join_operator_max_costs = np.ones(plan_type_counts[PlanType.JOIN])
        scan_operator_min_costs = np.ones(plan_type_counts[PlanType.SCAN])
        scan_operator_max_costs = np.ones(plan_type_counts[PlanType.SCAN])

        for plan_type, plan_index in plan_dict:
            plan = plan_dict[(plan_type, plan_index)]
            cost = cost_dict[plan]
            if plan_type == PlanType.JOIN:
                join_operator_min_costs[plan_index] = cost
                join_operator_max_costs[plan_index] = cost
            elif plan_type == PlanType.SCAN:
                scan_operator_min_costs[plan_index] = cost
                scan_operator_max_costs[plan_index] = cost
            else:
                raise ValueError("Unknown plan type")

        subgroup_costs = None
        if self._learn_decomposed_costs and len(subgroups) > 0:
            subgroup_cost_list = []
            for subgroup in subgroups:
                min_total_cost = subgroup_min_total_costs.get(subgroup, None)
                if min_total_cost is None:
                    subgroup_cost_list = None
                    break
                subgroup_cost_list.append(math.log(max(min_total_cost, 1e-10)))
            if subgroup_cost_list is not None:
                subgroup_costs = torch.FloatTensor(subgroup_cost_list)

        target_data = TopDownLearnedOptimizerTargetData(torch.log(torch.FloatTensor(join_operator_min_costs)),
                                                        torch.log(torch.FloatTensor(join_operator_max_costs)),
                                                        join_operator_cost_mask,
                                                        subgroup_costs,
                                                        torch.log(torch.FloatTensor(scan_operator_min_costs)),
                                                        torch.log(torch.FloatTensor(scan_operator_max_costs)),
                                                        scan_operator_cost_mask)

        training_data = TopDownLearnedOptimizerTrainingData(query_data, [target_data])
        return training_data

    def update_training_queries(self, training_queries: Set[SPJQuery]) -> bool:
        any_changes = False
        if len(training_queries) > len(self._last_training_queries) and self._feature_selected_sample_encoder is not None:
            changed = self._feature_selected_sample_encoder.update_feature_selected_sample_encoder(training_queries)
            any_changes = any_changes or changed
        self._last_training_queries = training_queries.copy()
        # Register the join columns of the training queries, so that a grown key column vocabulary counts as an encoder change.
        for query in training_queries:
            for join in query.joins():
                for _, column in join.equivalence_class():
                    self._key_column_index(column)
        if len(self._key_column_indexes) != self._last_num_key_columns:
            self._last_num_key_columns = len(self._key_column_indexes)
            any_changes = True
        if any_changes:
            self.version += 1
        return any_changes

    def _column_name_dicts(self) -> Tuple[Dict[Column, Tuple[str, str]], Dict[Tuple[str, str], Column]]:
        column_names = {}
        column_dict = {}
        for table, columns in self._column_orders.items():
            for column in columns:
                column_names[column] = (table.name(), column.name())
                column_dict[(table.name(), column.name())] = column
        return column_names, column_dict

    def _ordered_encoding_names(self, encodings: Dict[T, torch.FloatTensor]) -> List[str]:
        # The position of the one is the index the module was trained with, stored by name.
        names: List[Optional[str]] = [None] * len(encodings)
        for encoded_object, encoding in encodings.items():
            names[int(torch.argmax(encoding))] = str(encoded_object)
        return names

    def _encoding_order(self) -> Dict[str, List[str]]:
        table_names: List[Optional[str]] = [None] * len(self._table_index)
        for table, index in self._table_index.items():
            table_names[index] = table.name()
        return {"tables": table_names,
                "join_operators": self._ordered_encoding_names(self._join_operator_encodings),
                "scan_operators": self._ordered_encoding_names(self._scan_operator_encodings),
                "requirements": self._ordered_encoding_names(self._requirement_encodings)}

    def _apply_encoding_order(self, encoding_order: Dict[str, List[str]]):
        """Restores the one-hot index assignment the module was trained with, so that its weights
        keep referring to the tables and operators they were trained on."""
        tables_by_name = {table.name(): table for table in self._table_index}
        if sorted(encoding_order["tables"]) != sorted(tables_by_name):
            raise ValueError("The stored tables %s do not match the schema tables %s"
                             % (sorted(encoding_order["tables"]), sorted(tables_by_name)))
        self._table_encodings, self._table_index = self._build_one_hot_encodings(
            [tables_by_name[table_name] for table_name in encoding_order["tables"]])
        self._join_operator_encodings, _ = self._build_one_hot_encodings(
            self._objects_in_stored_order(self._join_operator_encodings, encoding_order["join_operators"], "join operators"))
        self._scan_operator_encodings, _ = self._build_one_hot_encodings(
            self._objects_in_stored_order(self._scan_operator_encodings, encoding_order["scan_operators"], "scan operators"))
        self._requirement_encodings, _ = self._build_one_hot_encodings(
            self._objects_in_stored_order(self._requirement_encodings, encoding_order["requirements"], "requirements"))

    @staticmethod
    def _objects_in_stored_order(encodings: Dict[T, torch.FloatTensor], names: List[str], description: str) -> List[T]:
        objects_by_name = {str(encoded_object): encoded_object for encoded_object in encodings}
        if sorted(objects_by_name) != sorted(names):
            raise ValueError("The stored %s %s do not match the configured %s %s"
                             % (description, sorted(names), description, sorted(objects_by_name)))
        return [objects_by_name[name] for name in names]

    def save(self, path: str):
        if self._random_sample_encoder is not None:
            self._random_sample_encoder.save(path + "_random_sample_encoder.json")
        if self._feature_selected_sample_encoder is not None:
            self._feature_selected_sample_encoder.save(path + "_feature_selected_sample_encoder.json")
        column_names, _ = self._column_name_dicts()
        key_column_names = [None] * len(self._key_column_indexes)
        for column, index in self._key_column_indexes.items():
            key_column_names[index] = column_names[column]
        with open(path + "_key_columns.json", "w") as f:
            json.dump(key_column_names, f)
        with open(path + "_encoding_order.json", "w") as f:
            json.dump(self._encoding_order(), f)

    def load(self, path: str):
        if self._random_sample_encoder is not None:
            self._random_sample_encoder.load(path + "_random_sample_encoder.json")
        if self._feature_selected_sample_encoder is not None:
            self._feature_selected_sample_encoder.load(path + "_feature_selected_sample_encoder.json")
        encoding_order_path = path + "_encoding_order.json"
        if not os.path.exists(encoding_order_path):
            raise ValueError(
                "%s is missing. This checkpoint was written before the one-hot encoding order was persisted, and the "
                "order it was trained with cannot be reconstructed: it followed the identity hashes of that process. "
                "Loading it would feed the module permuted table and operator features, so it has to be retrained."
                % encoding_order_path)
        with open(encoding_order_path) as f:
            self._apply_encoding_order(json.load(f))
        key_columns_path = path + "_key_columns.json"
        if os.path.exists(key_columns_path):
            with open(key_columns_path) as f:
                key_column_names = json.load(f)
            _, column_dict = self._column_name_dicts()
            self._key_column_indexes = {column_dict[(table_name, column_name)]: i for i, (table_name, column_name) in enumerate(key_column_names)}
            self._last_num_key_columns = len(self._key_column_indexes)



