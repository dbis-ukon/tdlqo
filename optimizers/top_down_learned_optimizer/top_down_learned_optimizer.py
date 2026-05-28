import datetime
import json
import logging
import math
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union

import binpacking
import torch
import numpy as np
import torch_geometric
import torch_scatter
import gc

from cardinality_estimators.cardinality_estimator import CardinalityMode
from cardinality_estimators.cardinality_range import CardinalityRange
from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from execution_engines.execution_data import ExecutionData
from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from execution_engines.postgresql_execution_engine import PostgreSQLExecutionEngine
from neural_network_modules.utility import print_parameters
from optimizers.top_down_cost_based_optimizer import TopDownCostBasedOptimizer
from optimizers.top_down_learned_optimizer.memory.memoized_expression_memory import MemoizedExpressionMemory
from optimizers.top_down_learned_optimizer.memory.group_memory import GroupMemory
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration, TrainingQuerySelectionMode
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import \
    TopDownLearnedOptimizerData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_calculation_data import \
    TopDownLearnedOptimizerTargetCalculationData, TopDownLearnedOptimizerTargetCalculationCostData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import \
    PlanType, TopDownLearnedOptimizerTargetData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_training_data import \
    TopDownLearnedOptimizerTrainingData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_encoder import TopDownLearnedOptimizerEncoder
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_module import TopDownLearnedOptimizerModule
from optimizers.top_down_optimizer import TopDownOptimizer
from queries.predicates.join import Join
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from schemas.schema import Schema
import concurrent.futures

class TopDownLearnedOptimizer(TopDownOptimizer):
    def __init__(self, schema: Schema, configuration: TopDownLearnedOptimizerConfiguration):
        self._schema = schema
        self._training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._inference_device = torch.device("cpu")
        super().__init__("Top-Down Learned Optimizer", configuration.name, configuration.description, configuration.available_operators)
        self._encoder = TopDownLearnedOptimizerEncoder(configuration, schema)
        self._double_q_learning = configuration.double_q_learning
        if configuration.double_q_learning:
            self._module_a = TopDownLearnedOptimizerModule(configuration, self._encoder)
            self._module_b = TopDownLearnedOptimizerModule(configuration, self._encoder)
            self._optimizer_a = torch.optim.Adam(self._module_a.parameters(), lr=configuration.learning_rate, weight_decay=configuration.weight_decay)
            self._optimizer_b = torch.optim.Adam(self._module_b.parameters(), lr=configuration.learning_rate, weight_decay=configuration.weight_decay)
        else:
            self._module = TopDownLearnedOptimizerModule(configuration, self._encoder)
            self._optimizer = torch.optim.Adam(self._module.parameters(), lr=configuration.learning_rate, weight_decay=configuration.weight_decay)

        self._untrained_postgresql = configuration.untrained_postgresql

        self._pre_training_seconds = configuration.pre_training_seconds
        self._exploration_timeout_seconds = configuration.exploration_timeout_seconds

        self._learn_decomposed_costs = configuration.learn_decomposed_costs
        self._min_local_cost = configuration.min_local_cost

        self._canonical_queries: Dict[SPJQuery, SPJQuery] = {}
        self._training_table_occurrence_encodings = {}

        self._deduce_cardinality_ranges = configuration.deduce_cardinality_ranges
        self._composite_max_degree_scaling = configuration.composite_max_degree_scaling
        self._shrinking_bounds_computed: Set[SPJQuery] = set()
        self._probed_subplan_queries: Set[SPJQuery] = set()
        self._upper_bounds: Dict[SPJQuery, Dict[SPJQuery, float]] = {}
        self._lower_bounds: Dict[SPJQuery, Dict[SPJQuery, float]] = {}

        self._training_query_selection_mode = configuration.training_query_selection_mode

        self._fully_explored_supervised = configuration.fully_explored_supervised

        self._module_to(self._inference_device)
        self._recursive_cost_constraint = configuration.recursive_cost_constraint_factor is not None
        self._num_training_epochs = configuration.num_training_epochs
        self._current_num_training_epochs = configuration.num_training_epochs if configuration.initial_training_epochs is None else configuration.initial_training_epochs
        self._training_epoch_decay = configuration.training_epoch_decay
        self._max_batch_size = configuration.max_batch_size
        self._target_batch_size = configuration.target_batch_size
        self.cost_model = configuration.cost_model
        cardinality_estimator = self.cost_model.cardinality_estimator
        assert isinstance(cardinality_estimator, PropheticCardinalityEstimator)
        self._cardinality_estimator: PropheticCardinalityEstimator = cardinality_estimator

        self._probe_min_cardinality_threshold = configuration.probe_min_cardinality_threshold
        self._probe_max_min_ratio_threshold = configuration.probe_max_min_ratio_threshold
        self._probe_per_probe_time_budget = configuration.probe_per_probe_time_budget
        self._probe_fetch_size = configuration.probe_fetch_size

        self._workers = configuration.workers

        self._training_queries: Set[SPJQuery] = set()
        self._subquery_origins: Dict[SPJQuery, SPJQuery] = {}
        self._new_cardinalities: Set[SPJQuery] = set()
        self._memory_queries: Set[SPJQuery] = set()
        self._memory: Dict[GroupRelationalAlgebraExpression, GroupMemory] = {}
        self._memory_groups: Dict[SPJQuery, Set[GroupRelationalAlgebraExpression]] = {}
        self._group_to_target_encoding_index: Dict[GroupRelationalAlgebraExpression, int] = {}
        self._target_encodings: List[TopDownLearnedOptimizerData] = []
        self._cardinalities_added_to_memory = 0

        self.trained = False

    def _module_to(self, device: torch.device):
        if self._double_q_learning:
            self._module_a.to(device)
            self._module_b.to(device)
        else:
            self._module.to(device)

    def _init_optimization(self, query: Query):
        assert isinstance(query, SPJQuery)
        self._table_occurrence_encodings = self._encoder.encode_table_occurrences(query)
        self._module_to(self._inference_device)

    def _choose(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> Optional[RelationalAlgebraExpression]:
        if self._untrained_postgresql and not self.trained:
            return None
        assert len(memoized_expressions) > 0
        if len(memoized_expressions) == 1:
            return memoized_expressions[0]
        min_cost, best_plan = self._choose_base(group, memoized_expressions)
        return best_plan

    def _choose_base(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> Tuple[float, RelationalAlgebraExpression]:
        query_data, plan_dict, _ = self._encoder.encode(group, memoized_expressions, table_occurrence_encodings=self._table_occurrence_encodings)
        if self._double_q_learning:
            module = random.choice([self._module_a, self._module_b])
            module_name = "A" if module is self._module_a else "B"
        else:
            module = self._module
            module_name = "main"
        with torch.no_grad():
            join_action_costs, scan_action_costs = module.forward(query_data)
        debug_logger = getattr(self, "_debug_logger", None)
        min_cost = None
        best_plan = None
        join_action_costs_np = join_action_costs.numpy()
        scan_action_costs_np = scan_action_costs.numpy()
        for i, join_action_cost in enumerate(join_action_costs_np):
            if min_cost is None or join_action_cost < min_cost:
                min_cost = join_action_cost
                best_plan = plan_dict[(PlanType.JOIN, i)]
        for i, scan_action_cost in enumerate(scan_action_costs_np):
            if min_cost is None or scan_action_cost < min_cost:
                min_cost = scan_action_cost
                best_plan = plan_dict[(PlanType.SCAN, i)]
        assert best_plan is not None

        if debug_logger is not None:
            def _cost_model_bounds_str(plan):
                try:
                    max_lc = self.cost_model.local_cost(plan, cardinality_mode=CardinalityMode.MAX)
                    min_lc = self.cost_model.local_cost(plan, cardinality_mode=CardinalityMode.MIN)
                except Exception as e:
                    return "cost_model_min=ERR:%s cost_model_max=ERR:%s" % (type(e).__name__, type(e).__name__)
                # Mirror the swap _compute_costs does when the cost formula is non-monotonic in cardinality.
                if max_lc is not None and min_lc is not None and min_lc > max_lc:
                    min_lc, max_lc = max_lc, min_lc
                min_str = "%.6f" % min_lc if min_lc is not None else "None"
                max_str = "%.6f" % max_lc if max_lc is not None else "None"
                return "cost_model_min=%s cost_model_max=%s" % (min_str, max_str)

            debug_logger.info("TDL decide group: %s (module=%s)" % (group.string(), module_name))
            for i, join_action_cost in enumerate(join_action_costs_np):
                plan = plan_dict[(PlanType.JOIN, i)]
                debug_logger.info("  plan=%s action_cost=%.6f %s" % (plan.string(), float(join_action_cost), _cost_model_bounds_str(plan)))
            for i, scan_action_cost in enumerate(scan_action_costs_np):
                plan = plan_dict[(PlanType.SCAN, i)]
                debug_logger.info("  plan=%s action_cost=%.6f %s" % (plan.string(), float(scan_action_cost), _cost_model_bounds_str(plan)))
            debug_logger.info("  chosen=%s cost=%.6f" % (best_plan.string(), float(min_cost)))

        return min_cost, best_plan

    def get_costs(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> Dict[RelationalAlgebraExpression, float]:
        query_data, plan_dict, _ = self._encoder.encode(group, memoized_expressions)
        if self._double_q_learning:
            module = random.choice([self._module_a, self._module_b])
        else:
            module = self._module
        with torch.no_grad():
            join_action_costs, scan_action_costs = module.forward(query_data)
        costs: Dict[RelationalAlgebraExpression, float] = {}
        for i, join_action_cost in enumerate(join_action_costs.numpy()):
            costs[plan_dict[(PlanType.JOIN, i)]] = float(join_action_cost)
        for i, scan_action_cost in enumerate(scan_action_costs.numpy()):
            costs[plan_dict[(PlanType.SCAN, i)]] = float(scan_action_cost)
        return costs

    def _canonicalize_query(self, query: SPJQuery) -> SPJQuery:
        canonical_query = self._canonical_queries.get(query, None)
        if canonical_query is None:
            self._canonical_queries[query] = query
            canonical_query = query
        return canonical_query

    def add_to_training_data(self, execution_data: ExecutionData):
        self._training_queries.add(execution_data.query)
        self._add_to_training_data_base(execution_data)
        self._probe_subplan_lower_bounds(execution_data)

    def _probe_subplan_lower_bounds(self, execution_data: ExecutionData) -> None:
        if execution_data.plan is None or execution_data.timeout is None:
            return
        candidates = []
        for subplan_query in execution_data.plan.subplan_queries():
            if not isinstance(subplan_query, SPJQuery):
                continue
            if len(subplan_query.table_occurrences()) < 2:
                continue
            if not subplan_query.is_connected():
                continue
            if subplan_query in self._probed_subplan_queries:
                continue
            candidates.append(subplan_query)
        candidates.sort(key=lambda q: len(q.table_occurrences()))
        for subplan_query in candidates:
            existing = self._cardinality_estimator.memory.get(subplan_query)
            if existing is not None:
                if (self._probe_min_cardinality_threshold is not None
                        and existing.min_cardinality >= self._probe_min_cardinality_threshold):
                    continue
                if (existing.min_cardinality > 0
                        and existing.max_cardinality is not None
                        and existing.max_cardinality / existing.min_cardinality <= self._probe_max_min_ratio_threshold):
                    continue
            self._probed_subplan_queries.add(subplan_query)
            cardinality_range = self._probe_subplan_lower_bound(subplan_query)
            if cardinality_range is not None:
                self._update_cardinality_ranges(subplan_query, cardinality_range)

    def _probe_subplan_lower_bound(self, query: SPJQuery) -> Optional[CardinalityRange]:
        timeout_ms = int(self._probe_per_probe_time_budget * 1000) + 500
        connection = self._schema.connection()
        rows_fetched = 0
        drained = False
        server_cursor = None
        try:
            with connection.cursor() as setup_cursor:
                setup_cursor.execute("SET statement_timeout = %d;" % timeout_ms)
            server_cursor = connection.cursor(name="probe_cursor_%d" % id(query))
            server_cursor.itersize = self._probe_fetch_size
            server_cursor.execute(query.get_query_text("1"))
            deadline = time.time() + self._probe_per_probe_time_budget
            while time.time() < deadline:
                rows = server_cursor.fetchmany(self._probe_fetch_size)
                rows_fetched += len(rows)
                if len(rows) < self._probe_fetch_size:
                    drained = True
                    break
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
        finally:
            if server_cursor is not None:
                try:
                    server_cursor.close()
                except Exception:
                    pass
            try:
                connection.close()
            except Exception:
                pass
        if rows_fetched == 0 and not drained:
            return None
        if drained:
            return CardinalityRange(rows_fetched, rows_fetched)
        return CardinalityRange(rows_fetched, None)

    def _add_to_training_data_base(self, execution_data: ExecutionData, original_query: Optional[SPJQuery] = None):
        if original_query is None:
            original_query = execution_data.query
        if (original_query not in self._subquery_origins) and (self._recursive_cost_constraint or self._deduce_cardinality_ranges):
            subplan_queries = self.enumerate_logical_groups(original_query)
            for subplan_query in subplan_queries:
                self._subquery_origins[subplan_query] = original_query
            if self._recursive_cost_constraint:
                assert isinstance(original_query, SPJQuery)
                for group_query in subplan_queries:
                    if group_query not in self._memory_queries:
                        self._new_cardinalities.add(group_query)
            if self._deduce_cardinality_ranges:
                self._add_bounds(original_query, subplan_queries)
        for query, (cardinality_range, true_subquery) in execution_data.cardinality_ranges.items():
            assert isinstance(query, SPJQuery)
            if query.is_connected():
                self._cardinalities_added_to_memory += 1
                if self._deduce_cardinality_ranges and true_subquery:
                    if cardinality_range.max_cardinality == 0:
                        self._deduce_zero_cardinalities(original_query, query)
                self._update_cardinality_ranges(query, cardinality_range)
                self._subquery_origins[query] = original_query
        if execution_data.timeout is not None and execution_data.plan is not None and self._training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED:
            for subplan_query in execution_data.plan.subplan_queries():
                if subplan_query not in self._memory_queries:
                    assert isinstance(subplan_query, SPJQuery)
                    self._new_cardinalities.add(subplan_query)

    def _deduce_zero_cardinalities(self, query: SPJQuery, subplan_query: SPJQuery):
        if subplan_query in self._cardinality_estimator.memory and self._cardinality_estimator.memory[subplan_query].max_cardinality == 0:
            return
        query_table_occurrences = subplan_query.table_occurrences()
        original_query_join_graph_edge_dict = query.join_graph_edge_dict()
        for other_table_occurrence in query.table_occurrences():
            if other_table_occurrence in query_table_occurrences:
                continue
            found_connection = False
            for table_occurrence, _ in original_query_join_graph_edge_dict[other_table_occurrence]:
                if table_occurrence in query_table_occurrences:
                    found_connection = True
                    break
            if not found_connection:
                continue
            expanded_query = query.induced_subquery(list(query_table_occurrences) + [other_table_occurrence])
            self._update_cardinality_ranges(expanded_query, CardinalityRange.exact_cardinality_range(0))
            self._subquery_origins[expanded_query] = query
            self._deduce_zero_cardinalities(query, expanded_query)

    def _add_bounds(self, query: SPJQuery, subplan_queries: List[SPJQuery]):
        for subplan_query in subplan_queries:
            if subplan_query in self._shrinking_bounds_computed:
                continue
            self._add_bounds_shrinking(subplan_query)

    def _add_bounds_shrinking(self, subplan_query: SPJQuery):
        if subplan_query not in self._lower_bounds:
            self._lower_bounds[subplan_query] = {}
            self._upper_bounds[subplan_query] = {}
        subplan_query_join_graph_edge_dict = subplan_query.join_graph_edge_dict()
        self._shrinking_bounds_computed.add(subplan_query)
        for table_occurrence in subplan_query.table_occurrences():
            edges = subplan_query_join_graph_edge_dict[table_occurrence]
            joins = set(join for _, join in edges)
            table = table_occurrence.table()
            # Collect all columns of this table that participate in equi-joins,
            # and build a mapping from each local column to its join partners.
            joined_columns = set()
            join_partners = {}
            for join in joins:
                local_columns = []
                other_columns = []
                for eq_table_occurrence, eq_column in join.equivalence_class():
                    if eq_table_occurrence == table_occurrence:
                        local_columns.append(eq_column)
                    else:
                        other_columns.append((eq_table_occurrence, eq_column))
                for col in local_columns:
                    joined_columns.add(col)
                    if col not in join_partners:
                        join_partners[col] = set()
                    for other in other_columns:
                        join_partners[col].add(other)
            # Determine the edge scale: 1.0 when a unique index of the removed table is covered
            # by the join columns, else max_deg_T(joined_columns) fetched lazily from the cost
            # model (tight: group size on the full joined-column set of T, NULLs excluded).
            covering_index = None
            for index in table.indexes():
                if index.unique() and joined_columns.issuperset(frozenset(index.columns())):
                    covering_index = index
                    break
            if covering_index is not None:
                scale = 1.0
            else:
                if not self._composite_max_degree_scaling:
                    continue
                if not joined_columns:
                    continue
                composite_max_degree = self.cost_model.composite_max_degree(table, joined_columns)
                if composite_max_degree is None:
                    continue
                scale = float(composite_max_degree)
            other_table_occurrences = [t for t in subplan_query.table_occurrences() if t != table_occurrence]
            shrunk_query = subplan_query.induced_subquery(other_table_occurrences)
            if not shrunk_query.is_connected():
                continue
            if shrunk_query not in self._lower_bounds:
                self._lower_bounds[shrunk_query] = {}
                self._upper_bounds[shrunk_query] = {}
            # Exact cardinality bound requires: (1) no predicate on the removed table,
            # (2) a foreign key from a single neighboring table that covers the unique
            # index, with NOT NULL FK columns, (3) the join conditions match the
            # FK column mapping, and (4) the removed table has no joins on columns
            # outside the covering index (otherwise removing it drops those conditions).
            exact_cardinality_bound = False
            if covering_index is not None:
                covering_columns = frozenset(covering_index.columns())
                if isinstance(table_occurrence.predicate(), TruePredicate) and covering_columns.issuperset(joined_columns):
                    # Collect distinct neighboring table occurrences from join partners.
                    neighbor_table_occurrences = set()
                    for partners in join_partners.values():
                        for other_to, _ in partners:
                            neighbor_table_occurrences.add(other_to)
                    for neighbor_to in neighbor_table_occurrences:
                        neighbor_table = neighbor_to.table()
                        for fk in self._schema.foreign_keys():
                            if fk.foreign_key_table() != neighbor_table or fk.primary_key_table() != table:
                                continue
                            fk_mapping = fk.mapping()
                            pk_cols_in_fk = set(fk_mapping.values())
                            if not pk_cols_in_fk.issuperset(covering_columns):
                                continue
                            reverse_mapping = {pk_col: fk_col for fk_col, pk_col in fk_mapping.items()}
                            if not all(not reverse_mapping[pk_col].nullable() for pk_col in covering_columns):
                                continue
                            if all((neighbor_to, reverse_mapping[pk_col]) in join_partners.get(pk_col, set()) for pk_col in covering_columns):
                                exact_cardinality_bound = True
                                break
                        if exact_cardinality_bound:
                            break
            self._save_bounds(subplan_query, shrunk_query, exact_cardinality_bound, scale=scale)

    def _save_bounds(self, smaller_cardinality_query: SPJQuery, larger_cardinality_query: SPJQuery, equal: bool, scale: float = 1.0, verify: bool = False):
        # Edge semantics: |smaller| <= |larger| * scale.
        #   _upper_bounds[larger][smaller] = c  ->  max(smaller) <= max(larger) * c
        #   _lower_bounds[smaller][larger] = c  ->  min(larger)  >= min(smaller) / c
        # Legacy scale-1 containment edges still fit (multiply/divide by 1 are no-ops).
        assert scale > 0
        assert not (equal and scale != 1.0)
        existing_upper = self._upper_bounds[larger_cardinality_query].get(smaller_cardinality_query)
        if existing_upper is None or scale < existing_upper:
            self._upper_bounds[larger_cardinality_query][smaller_cardinality_query] = scale
        existing_lower = self._lower_bounds[smaller_cardinality_query].get(larger_cardinality_query)
        if existing_lower is None or scale < existing_lower:
            self._lower_bounds[smaller_cardinality_query][larger_cardinality_query] = scale
        if equal:
            self._lower_bounds[larger_cardinality_query][smaller_cardinality_query] = 1.0
            self._upper_bounds[smaller_cardinality_query][larger_cardinality_query] = 1.0
        if verify:
            cursor = self._schema.connection().cursor()
            cursor.execute(smaller_cardinality_query.query_text())
            smaller_cardinality = cursor.fetchone()[0]
            cursor.execute(larger_cardinality_query.query_text())
            larger_cardinality = cursor.fetchone()[0]
            if smaller_cardinality > larger_cardinality * scale or (equal and smaller_cardinality != larger_cardinality):
                print("Cardinality bounds violated:")
                print("Equal: %s" % equal)
                print("Scale: %.3f" % scale)
                print("Smaller cardinality query: %s" % smaller_cardinality_query.query_text())
                print("Larger cardinality query: %s" % larger_cardinality_query.query_text())
                print("Cardinalities: %d > %d * %.3f" % (smaller_cardinality, larger_cardinality, scale))
                raise ValueError("Cardinality bounds violated.")
            cursor.close()
        if smaller_cardinality_query in self._cardinality_estimator.memory:
            self._update_cardinality_range_lower_bound(larger_cardinality_query, self._cardinality_estimator.memory[smaller_cardinality_query], scale=scale)
        if larger_cardinality_query in self._cardinality_estimator.memory:
            self._update_cardinality_range_upper_bound(smaller_cardinality_query, self._cardinality_estimator.memory[larger_cardinality_query], scale=scale)

    def _update_cardinality_range_lower_bound(self, query: SPJQuery, cardinality_range: CardinalityRange, scale: float = 1.0):
        if scale == 0:
            return
        new_min = math.floor(cardinality_range.min_cardinality / scale)
        lower_bound_range = CardinalityRange(new_min, None)
        self._update_cardinality_ranges(query, lower_bound_range)

    def _update_cardinality_range_upper_bound(self, query: SPJQuery, cardinality_range: CardinalityRange, scale: float = 1.0):
        if cardinality_range.max_cardinality is None:
            return
        new_max = math.ceil(cardinality_range.max_cardinality * scale)
        upper_bound_range = CardinalityRange(0, new_max)
        self._update_cardinality_ranges(query, upper_bound_range)

    def _update_cardinality_ranges(self, query: SPJQuery, cardinality_range: CardinalityRange):
        if cardinality_range.has_no_information():
            return
        query = self._canonicalize_query(query)
        min_cardinality_changed, max_cardinality_changed = self._cardinality_estimator.memorize(query, cardinality_range)
        if query not in self._memory_queries:
            if self._training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER:
                add_query = max_cardinality_changed
            elif self._training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER or self._training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED:
                add_query = max_cardinality_changed or min_cardinality_changed
            else:
                raise NotImplementedError
            if add_query:
                self._new_cardinalities.add(query)
        if self._deduce_cardinality_ranges:
            if min_cardinality_changed:
                for lower_bound_query, scale in list(self._lower_bounds.get(query, {}).items()):
                    self._update_cardinality_range_lower_bound(lower_bound_query, cardinality_range, scale=scale)
            if max_cardinality_changed:
                for upper_bound_query, scale in list(self._upper_bounds.get(query, {}).items()):
                    self._update_cardinality_range_upper_bound(upper_bound_query, cardinality_range, scale=scale)

    def _get_possible_groups(self, query: SPJQuery) -> List[GroupRelationalAlgebraExpression]:
        groups = [GroupRelationalAlgebraExpression(query, Requirements())]
        table_occurrences = list(query.table_occurrences())
        if len(table_occurrences) > 1 or IndexScanExpression not in self.available_operators:
            return groups
        table_occurrence = table_occurrences[0]
        for index in table_occurrence.table().indexes():
            groups.append(GroupRelationalAlgebraExpression(query, Requirements(force_index_scan=(index, True))))
            groups.append(GroupRelationalAlgebraExpression(query, Requirements(force_index_scan=(index, False))))
        return groups

    def _add_group_to_memory(self, group: GroupRelationalAlgebraExpression, original_query: SPJQuery):
        memoized_expressions = self.enumerate_memoized_expressions(group)
        for memoized_expression in memoized_expressions:
            memoized_expression.canonicalize(self._canonical_queries)
        self._add_group_to_memory_base(group, memoized_expressions, original_query)

    def _add_group_to_memory_base(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression], original_query: SPJQuery):
        query_encoding, plan_dict, subplan_queries = self._encoder.encode(group, memoized_expressions, table_occurrence_encodings=self._training_table_occurrence_encodings)
        memoized_expression_memories = []
        for plan_type, plan_index in plan_dict:
            memoized_expression = plan_dict[(plan_type, plan_index)]
            memoized_expression_memory = MemoizedExpressionMemory(memoized_expression, plan_type, plan_index)
            memoized_expression_memories.append(memoized_expression_memory)
            if self._recursive_cost_constraint and not self._learn_decomposed_costs:
                self._compute_target_memory_indexes(memoized_expression_memory)
        if self._learn_decomposed_costs:
            subgroup_target_memory_indexes = self._compute_subgroup_target_memory_indexes(subplan_queries)
        else:
            subgroup_target_memory_indexes = []
        query_memory = GroupMemory(query_encoding, memoized_expression_memories, subgroup_target_memory_indexes)
        self._memory[group] = query_memory
        if original_query not in self._memory_groups:
            self._memory_groups[original_query] = set()
        self._memory_groups[original_query].add(group)

    def _compute_target_memory_indexes(self, memoized_expression_memory: MemoizedExpressionMemory):
        # If we use the recursive cost constraint, we compute this as soon as we add a memoized expression to the memory, since we use all memoized expressions during training.
        # If we do not use the recursive cost constraint, we only compute this once we have a local cost for the memoized expression, since we only use the memoized expressions with a local cost during training.
        target_memory_indexes = []
        for child_group in memoized_expression_memory.memoized_expression.children:
            assert isinstance(child_group, GroupRelationalAlgebraExpression)
            target_memory_index = self._group_to_target_encoding_index.get(child_group, None)
            if target_memory_index is None:
                group_memory = self._memory.get(child_group, None)
                if group_memory is None:
                    group_query_data, _, _ = self._encoder.encode(child_group, self.enumerate_memoized_expressions(child_group), table_occurrence_encodings=self._training_table_occurrence_encodings)
                else:
                    group_query_data = self._memory[child_group].encoding
                target_memory_index = len(self._target_encodings)
                self._group_to_target_encoding_index[child_group] = target_memory_index
                self._target_encodings.append(group_query_data)
            target_memory_indexes.append(target_memory_index)
        memoized_expression_memory.target_memory_indexes = target_memory_indexes

    def _compute_subgroup_target_memory_indexes(self, subgroups: List[GroupRelationalAlgebraExpression]) -> List[int]:
        target_memory_indexes = []
        for subgroup in subgroups:
            target_memory_index = self._group_to_target_encoding_index.get(subgroup, None)
            if target_memory_index is None:
                group_memory = self._memory.get(subgroup, None)
                if group_memory is None:
                    subgroup_data, _, _ = self._encoder.encode(subgroup, self.enumerate_memoized_expressions(subgroup), table_occurrence_encodings=self._training_table_occurrence_encodings)
                else:
                    subgroup_data = self._memory[subgroup].encoding
                target_memory_index = len(self._target_encodings)
                self._group_to_target_encoding_index[subgroup] = target_memory_index
                self._target_encodings.append(subgroup_data)
            target_memory_indexes.append(target_memory_index)
        return target_memory_indexes

    def explore(self, queries: List[Query], time_budget: Optional[float], logger: Optional[logging.Logger] = None):
        start_time = time.time()
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        logger.info("Exploring %d queries - %s" % (len(queries), datetime.datetime.utcnow().isoformat()))
        spj_queries = [query for query in queries if isinstance(query, SPJQuery)]
        for spj_query in spj_queries:
            self._training_queries.add(spj_query)

        @dataclass
        class QueryExploration:
            query: SPJQuery
            groups: List[SPJQuery]

        logger.info("Enumerating groups for %d queries." % len(spj_queries))
        def process_query(query):
            query_groups = TopDownLearnedOptimizer.enumerate_logical_groups(query)
            query_groups.sort(key=lambda q: -len(q.table_occurrences()))
            return QueryExploration(query, query_groups), query_groups

        all_logical_groups = set()
        query_explorations = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) as executor:
            results = list(executor.map(process_query, spj_queries))

        for query_exploration, query_groups in results:
            query_explorations.append(query_exploration)
            all_logical_groups.update(query_groups)

        logger.info("Enumerated %d distinct groups." % len(all_logical_groups))

        query_queue = queue.SimpleQueue()
        while any(query_explorations):
            for query_exploration in query_explorations:
                if len(query_exploration.groups) > 0:
                    query_queue.put((query_exploration.query, query_exploration.groups.pop(0)))
            query_explorations = [query_exploration for query_exploration in query_explorations if len(query_exploration.groups) > 0]

        additional_set_commands = ["SET enable_bitmapscan = off;",
                                   "SET enable_mergejoin = off;"]
        execution_count = 0
        lock = threading.Lock()

        def execute_query(args: Tuple[SPJQuery, SPJQuery, Optional[float]]) -> ExecutionData:
            query, group, end_time = args
            nonlocal execution_count
            if group in self._cardinality_estimator.memory:
                cardinality_range = self._cardinality_estimator.memory[group]
                if cardinality_range.is_exact():
                    return None

            if end_time is None and self._exploration_timeout_seconds is None:
                timeout = None
            elif end_time is None and self._exploration_timeout_seconds is not None:
                timeout = self._exploration_timeout_seconds * 1000
            elif end_time is not None and self._exploration_timeout_seconds is None:
                timeout = int((end_time - time.time()) * 1000)
            else:
                timeout = int(min(end_time - time.time(), self._exploration_timeout_seconds) * 1000)

            if timeout is not None and timeout <= 0:
                timeout = 1

            execution_engine = PostgreSQLExecutionEngine(self._schema, 0, timeout=timeout, additional_set_commands=additional_set_commands)
            execution_data = execution_engine.execute(group, None)
            with lock:
                execution_count += 1
                self._add_to_training_data_base(execution_data, original_query=query)
                if execution_data.timeout is not None:
                    logger.info("Execution %d timed out." % execution_count)
                else:
                    logger.info("Execution %d completed - %d cardinalities in memory." % (execution_count, len(self._cardinality_estimator.memory)))
            return execution_data

        if time_budget is not None:
            end_time = start_time + time_budget
        else:
            end_time = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) as executor:
            futures = []
            while not query_queue.empty():
                futures.append(executor.submit(execute_query, (*query_queue.get(), end_time)))
            try:
                for f in concurrent.futures.as_completed(futures):
                    time_elapsed = time.time() - start_time
                    if time_budget is not None and time_elapsed >= time_budget:
                        break
            except concurrent.futures.TimeoutError:
                logger.info("Time budget reached, cancelling remaining tasks.")

            # Cancel any unfinished futures
            for f in futures:
                if not f.done():
                    f.cancel()

        logger.info("Exploration finished - %s" % datetime.datetime.utcnow().isoformat())
        return len(all_logical_groups) == len([cardinality_range for cardinality_range in self._cardinality_estimator.memory.values() if cardinality_range.is_exact()])

    def explore_and_train(self, queries: List[Query], time_budget: Optional[float], logger: Optional[logging.Logger] = None, start_time: Optional[float] = None, path: Optional[str] = None) -> Tuple[bool, dict]:
        start_time = time.time()
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        logger.info("Exploring %d queries - %s" % (len(queries), datetime.datetime.utcnow().isoformat()))
        spj_queries = [query for query in queries if isinstance(query, SPJQuery)]
        for spj_query in spj_queries:
            self._training_queries.add(spj_query)

        @dataclass
        class QueryExploration:
            query: SPJQuery
            groups: List[SPJQuery]

        # --- 1. Initial Grouping (Uses standard workers) ---
        logger.info("Enumerating groups for %d queries." % len(spj_queries))

        def process_query(query):
            query_groups = TopDownLearnedOptimizer.enumerate_logical_groups(query)
            query_groups.sort(key=lambda q: -len(q.table_occurrences()))
            return QueryExploration(query, query_groups), query_groups

        all_logical_groups = set()
        query_explorations = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) as executor:
            results = list(executor.map(process_query, spj_queries))

        for query_exploration, query_groups in results:
            query_explorations.append(query_exploration)
            all_logical_groups.update(query_groups)

        logger.info("Enumerated %d distinct groups." % len(all_logical_groups))

        # --- 2. Setup Thresholds and Synchronization ---
        training_threshold_executions = 200
        training_threshold_cardinalities = min(10000, len(all_logical_groups) // 4)

        query_queue = queue.SimpleQueue()  # Queue for Exploration (Group execution)
        plan_queue = queue.LifoQueue()
        data_processing_queue = queue.SimpleQueue()  # Queue for Plan Execution (Full query execution)

        # Fill exploration queue
        while any(query_explorations):
            for query_exploration in query_explorations:
                if len(query_exploration.groups) > 0:
                    query_queue.put((query_exploration.query, query_exploration.groups.pop(0)))
            query_explorations = [query_exploration for query_exploration in query_explorations if len(query_exploration.groups) > 0]

        additional_set_commands = ["SET enable_bitmapscan = off;", "SET enable_mergejoin = off;"]

        # Shared state
        execution_count = 0
        lock = threading.Lock()
        start_training_event = threading.Event()
        stop_training_event = threading.Event()

        # --- 3. Worker Functions ---
        def manage_data_processing():
            """Dedicated thread to process execution data without blocking workers."""
            nonlocal execution_count
            while not stop_training_event.is_set() or not data_processing_queue.empty():
                try:
                    item = data_processing_queue.get(timeout=0.5)
                    execution_data, original_query = item

                    with lock:
                        execution_count += 1
                        self._add_to_training_data_base(execution_data, original_query=original_query)

                        if (execution_data.timeout is None and not start_training_event.is_set() and len(self._cardinality_estimator.memory) >= training_threshold_cardinalities and execution_count >= training_threshold_executions):
                            start_training_event.set()

                        time_elapsed = time.time() - start_time
                        if time_budget is not None and time_elapsed >= time_budget:
                            start_training_event.set()
                            stop_training_event.set()
                            return

                    logger.info(f"Processed execution %d - %d cardinalities in memory." % (execution_count, len(self._cardinality_estimator.memory)))
                except queue.Empty:
                    continue

        def execute_exploration_query(args: Tuple[SPJQuery, SPJQuery, Optional[float]]) -> ExecutionData:
            query, group, end_time = args
            nonlocal execution_count

            if group in self._cardinality_estimator.memory:
                cardinality_range = self._cardinality_estimator.memory[group]
                if cardinality_range.is_exact():
                    return None

            if end_time is None and self._exploration_timeout_seconds is None:
                timeout = None
            elif end_time is None and self._exploration_timeout_seconds is not None:
                timeout = self._exploration_timeout_seconds * 1000
            elif end_time is not None and self._exploration_timeout_seconds is None:
                timeout = int((end_time - time.time()) * 1000)
            else:
                timeout = int(min(end_time - time.time(), self._exploration_timeout_seconds) * 1000)
            if timeout is not None and timeout <= 0:
                timeout = 1

            execution_engine = PostgreSQLExecutionEngine(self._schema, 0, timeout=timeout, additional_set_commands=additional_set_commands)
            execution_data = execution_engine.execute(group, None)
            data_processing_queue.put((execution_data, query))

        meta_data = {}

        def manage_training_and_optimization() -> None:
            nonlocal meta_data

            # Wait for the initial threshold
            start_training_event.wait()
            iteration = 0

            while not stop_training_event.is_set():
                meta_data = self._train(logger=logger, lock=lock)
                iteration += 1

                if start_time is not None:
                    current_time = time.time()
                    meta_data["optimizer_training_time_seconds"] = current_time - start_time

                if path is not None:
                    training_meta_data_path = path + "_%d_meta_data.json" % iteration
                    with open(training_meta_data_path, "w") as f:
                        json.dump(meta_data, f, indent=4)
                    self.save(path + "_%d" % iteration)

                if stop_training_event.is_set():
                    break

                current_batch_plans = []
                randomized_queries = queries.copy()
                random.shuffle(randomized_queries)
                for query in randomized_queries:
                    if stop_training_event.is_set():
                        break
                    plan = self.optimize(query)
                    current_batch_plans.append((query, plan))
                for item in current_batch_plans:
                    plan_queue.put(item)

            meta_data = self._train(logger=logger, lock=lock)
            iteration += 1

            if start_time is not None:
                current_time = time.time()
                meta_data["optimizer_training_time_seconds"] = current_time - start_time

            if path is not None:
                training_meta_data_path = path + "_%d_meta_data.json" % iteration
                with open(training_meta_data_path, "w") as f:
                    json.dump(meta_data, f, indent=4)
                self.save(path + "_%d" % iteration)

        def manage_plan_execution():
            nonlocal execution_count
            pg_hint_plan_execution_engine = PgHintPlanExecutionEngine(self._schema, 0, timeout=self._exploration_timeout_seconds * 1000)

            while not stop_training_event.is_set():
                try:
                    query, plan = plan_queue.get(timeout=1)
                except queue.Empty:
                    continue

                execution_data = pg_hint_plan_execution_engine.execute(query, plan)
                data_processing_queue.put((execution_data, query))

        # --- 4. Main Execution Loop ---

        if time_budget is not None:
            end_time = start_time + time_budget
        else:
            end_time = None

        # self._workers for exploration + 1 for Training/Opt + 1 for Plan Execution
        pool_size = self._workers + 2

        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = []

            # Submit the long-running managers
            training_future = executor.submit(manage_training_and_optimization)
            data_future = executor.submit(manage_data_processing)
            plan_future = executor.submit(manage_plan_execution)

            futures.append(training_future)
            futures.append(data_future)
            futures.append(plan_future)

            # Submit the exploration tasks
            while not query_queue.empty():
                futures.append(executor.submit(execute_exploration_query, (*query_queue.get(), end_time)))

            try:
                for f in concurrent.futures.as_completed(futures):
                    if f.exception() is not None:
                        logger.error(f"Thread failed with exception: {f.exception()}")
                        raise f.exception()

                    time_elapsed = time.time() - start_time
                    if time_budget is not None and time_elapsed >= time_budget:
                        break
            except concurrent.futures.TimeoutError:
                logger.info("Time budget reached, cancelling remaining tasks.")
            except Exception as e:
                logger.error("Exception during exploration and training: %s" % str(e))
                for f in futures:
                    f.cancel()
                raise e

            # Signal threads to stop
            stop_training_event.set()

            # Cancel any pending exploration tasks
            for f in futures:
                if f is not training_future and not f.done():
                    f.cancel()

        logger.info("Exploration finished - %s" % datetime.datetime.utcnow().isoformat())
        return len(all_logical_groups) == len([cardinality_range for cardinality_range in self._cardinality_estimator.memory.values() if cardinality_range.is_exact()]), meta_data

    def pre_train(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        logger.info("Collecting pre-training data.")

        cost_model = self.cost_model.replace_cardinality_estimator(PostgreSQLCardinalityEstimator(self._schema, True))
        top_down_cost_based_optimizer = TopDownCostBasedOptimizer(cost_model, self.available_operators)
        training_query_iterators = [top_down_cost_based_optimizer.get_all_local_costs(query) for query in self._training_queries]

        training_set = []
        start_time = datetime.datetime.utcnow()
        timed_out = False

        for iterator in training_query_iterators:
            if timed_out:
                break
            for (group, local_costs), subgroup_min_total_costs in iterator:
                training_data = self._encoder.encode_pre_training(group, local_costs, subgroup_min_total_costs)
                if training_data is not None:
                    training_set.append(training_data)
                if (datetime.datetime.utcnow() - start_time).total_seconds() > self._pre_training_seconds:
                    timed_out = True
                    break

        if len(training_set) == 0:
            logger.info("No training data collected during pre-training.")
            return

        logger.info("Collected %d training samples for pre-training." % len(training_set))

        num_batches = self._num_batches(len(training_set))
        if num_batches > 1:
            random.shuffle(training_set)
            raw_batches = np.array_split(training_set, num_batches)
        else:
            raw_batches = [training_set]
        batches = []
        for memory_batch in raw_batches:
            batch = TopDownLearnedOptimizerTrainingData.unify(memory_batch)
            batches.append(([batch.input_data], batch.target_data))

        self._module_to(self._training_device)
        for epoch in range(self._current_num_training_epochs):
            self._training_epoch(epoch, batches, logger, differentiate_double=False)

        logger.info("Pre-training on device: %s" % self._training_device.type)

        self._module_to(self._inference_device)
        self.trained = True

    def _compute_costs(self) -> Tuple[int, Dict[GroupRelationalAlgebraExpression, float], Dict[str, int]]:
        local_costs_computed = 0
        local_costs_with_non_zero_lower = 0
        local_costs_with_finite_upper = 0
        local_costs_with_non_zero_lower_and_finite_upper = 0
        sorted_memory_queries = sorted(self._memory.keys(), key=lambda group: len(group.query().table_occurrences()))
        exact_costs: Dict[GroupRelationalAlgebraExpression, float] = {}
        for group in sorted_memory_queries:
            group_memory = self._memory[group]
            all_costs_exact = True
            min_exact_cost = None
            for memoized_expression_memory in group_memory.memoized_expression_memories:
                max_local_cost = self.cost_model.local_cost(memoized_expression_memory.memoized_expression, cardinality_mode=CardinalityMode.MAX)
                if max_local_cost is not None or self._training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER or TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED:
                    min_local_cost = self.cost_model.local_cost(memoized_expression_memory.memoized_expression, cardinality_mode=CardinalityMode.MIN)
                    has_non_zero_lower = min_local_cost is not None and min_local_cost > 0
                    has_finite_upper = max_local_cost is not None
                    if min_local_cost is None:
                        min_local_cost = 1e-10
                    # Admittedly a bit strange, but cost models are not necessarily monotonic w.r.t. cardinality estimates, so min cost can be higher than max cost.
                    if max_local_cost is not None and min_local_cost > max_local_cost:
                        memoized_expression_memory.max_local_cost = min_local_cost
                        memoized_expression_memory.min_local_cost = max_local_cost
                    else:
                        memoized_expression_memory.max_local_cost = max_local_cost
                        memoized_expression_memory.min_local_cost = min_local_cost
                    if max_local_cost is None or abs(max_local_cost - min_local_cost) > 1e-9:
                        all_costs_exact = False
                    elif self._fully_explored_supervised and all_costs_exact:
                        child_groups = memoized_expression_memory.memoized_expression.children
                        child_group_costs = []
                        for child_group in child_groups:
                            assert isinstance(child_group, GroupRelationalAlgebraExpression)
                            child_group_cost = exact_costs.get(child_group, None)
                            if child_group_cost is None:
                                all_costs_exact = False
                                break
                            child_group_costs.append(child_group_cost)
                        if all_costs_exact:
                            total_cost = min_local_cost + sum(child_group_costs)
                            if min_exact_cost is None or total_cost < min_exact_cost:
                                min_exact_cost = total_cost
                    local_costs_computed += 1
                    if has_non_zero_lower:
                        local_costs_with_non_zero_lower += 1
                    if has_finite_upper:
                        local_costs_with_finite_upper += 1
                    if has_non_zero_lower and has_finite_upper:
                        local_costs_with_non_zero_lower_and_finite_upper += 1
                    if memoized_expression_memory.min_local_cost < 0:
                        raise ValueError("Computed local costs are invalid: min: %f, max: %f" % (memoized_expression_memory.min_local_cost, memoized_expression_memory.max_local_cost))
                    if not self._recursive_cost_constraint and not self._learn_decomposed_costs:
                        self._compute_target_memory_indexes(memoized_expression_memory)
                else:
                    all_costs_exact = False
            if self._fully_explored_supervised and all_costs_exact:
                exact_costs[group] = min_exact_cost
        target_bound_counts = {
            "local_costs_with_non_zero_lower": local_costs_with_non_zero_lower,
            "local_costs_with_finite_upper": local_costs_with_finite_upper,
            "local_costs_with_non_zero_lower_and_finite_upper": local_costs_with_non_zero_lower_and_finite_upper,
        }
        return local_costs_computed, exact_costs, target_bound_counts

    def train(self, logger: Optional[logging.Logger] = None) -> dict:
        return self._train(logger=logger)

    def _train(self, logger: Optional[logging.Logger] = None, lock: Optional[threading.Lock] = None) -> dict:
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        # store parameters pre training
        if self._double_q_learning:
            self._module_checkpoint_a = self._module_a.state_dict()
            self._module_checkpoint_b = self._module_b.state_dict()
        else:
            self._module_checkpoint = self._module.state_dict()

        if lock is not None:
            lock.acquire()

        if not self.trained:
            self._add_statistical_cardinality_bounds()

        try:
            encoder_changed = self._encoder.update_training_queries(self._training_queries)
            if encoder_changed:
                logger.info("Encoder changed, recalculating encodings.")
                self._training_table_occurrence_encodings = {}
            logger.info("Encoding table occurrences.")
            for training_query in self._training_queries:
                self._encoder.add_table_occurrence_encodings(training_query, self._training_table_occurrence_encodings)
            if encoder_changed:
                for original_query, groups in self._memory_groups.items():
                    for group in groups:
                        group_memory = self._memory[group]
                        self._add_group_to_memory_base(group, list(group_memory.memoized_expression_dict.keys()), original_query)
                for target_group, target_encoding_index in self._group_to_target_encoding_index.items():
                    target_group_memory = self._memory.get(target_group, None)
                    if target_group_memory is None:
                        target_encoding, _, _ = self._encoder.encode(target_group, self.enumerate_memoized_expressions(target_group), table_occurrence_encodings=self._training_table_occurrence_encodings)
                    else:
                        target_encoding = target_group_memory.encoding
                    self._target_encodings[target_encoding_index] = target_encoding

            logger.info("Encoding %d new queries - %s" % (len(self._new_cardinalities), datetime.datetime.utcnow().isoformat()))
            new_cardinalities_sorted = sorted(self._new_cardinalities, key=lambda q: len(q.table_occurrences()))
            for query in new_cardinalities_sorted:
                original_query = self._subquery_origins[query]
                self._memory_queries.add(query)
                for group in self._get_possible_groups(query):
                    self._add_group_to_memory(group, original_query)
            self._new_cardinalities.clear()
            logger.info("Encoding complete - %s" % datetime.datetime.utcnow().isoformat())

            if not self.trained and self._pre_training_seconds > 0:
                self.pre_train(logger)

            self._module_to(self._training_device)
            logger.info("Training on device: %s" % self._training_device.type)

            has_lower_bound = 0
            has_upper_bound = 0
            has_both_bounds = 0
            has_range_smaller_10 = 0
            has_exact_cardinality = 0
            has_exact_zero_cardinality = 0
            for query in self._cardinality_estimator.memory:
                cardinality_range = self._cardinality_estimator.memory[query]
                if cardinality_range.min_cardinality > 0:
                    has_lower_bound += 1
                if cardinality_range.max_cardinality is not None:
                    has_upper_bound += 1
                    if cardinality_range.min_cardinality > 0:
                        has_both_bounds += 1
                    if cardinality_range.max_cardinality - cardinality_range.min_cardinality < 10:
                        has_range_smaller_10 += 1
                    if cardinality_range.max_cardinality == cardinality_range.min_cardinality:
                        has_exact_cardinality += 1
                        if cardinality_range.max_cardinality == 0:
                            has_exact_zero_cardinality += 1
        finally:
            if lock is not None:
                lock.release()

        logger.info("Computing costs for training.")
        local_costs_computed, exact_costs, target_bound_counts = self._compute_costs()
        logger.info("Cost computation complete - %s" % datetime.datetime.utcnow().isoformat())

        exact_cost_target_indexes: Dict[int, float] = {}
        for query, cost in exact_costs.items():
            target_index = self._group_to_target_encoding_index.get(query, None)
            if target_index is not None:
                exact_cost_target_indexes[target_index] = cost

        meta_data = {}
        logger.info("Total number of cardinalities gathered: %d" % self._cardinalities_added_to_memory)
        meta_data["cardinalities_gathered"] = self._cardinalities_added_to_memory
        logger.info("Total number of distinct cardinalities: %d" % len(self._cardinality_estimator.memory))
        meta_data["distinct_cardinalities"] = len(self._cardinality_estimator.memory)
        logger.info("Total number of non-zero lower limits: %d" % has_lower_bound)
        meta_data["non_zero_lower_limits"] = has_lower_bound
        logger.info("Total number of finite upper limits: %d" % has_upper_bound)
        meta_data["finite_upper_limits"] = has_upper_bound
        logger.info("Total number of non-zero lower limits and finite upper limits: %d" % has_both_bounds)
        meta_data["non_zero_lower_and_finite_upper_limits"] = has_both_bounds
        logger.info("Total number of cardinality ranges smaller 10: %d" % has_range_smaller_10)
        meta_data["cardinality_ranges_smaller_10"] = has_range_smaller_10
        logger.info("Total number of exact cardinalities: %d" % has_exact_cardinality)
        meta_data["exact_cardinalities"] = has_exact_cardinality
        logger.info("Total number of exact zero cardinalities: %d" % has_exact_zero_cardinality)
        meta_data["exact_zero_cardinalities"] = has_exact_zero_cardinality
        logger.info("Total number of local costs computed: %d" % local_costs_computed)
        meta_data["local_costs_computed"] = local_costs_computed
        logger.info("Local costs with non-zero lower: %d" % target_bound_counts["local_costs_with_non_zero_lower"])
        meta_data["local_costs_with_non_zero_lower"] = target_bound_counts["local_costs_with_non_zero_lower"]
        logger.info("Local costs with finite upper: %d" % target_bound_counts["local_costs_with_finite_upper"])
        meta_data["local_costs_with_finite_upper"] = target_bound_counts["local_costs_with_finite_upper"]
        logger.info("Local costs with non-zero lower and finite upper: %d" % target_bound_counts["local_costs_with_non_zero_lower_and_finite_upper"])
        meta_data["local_costs_with_non_zero_lower_and_finite_upper"] = target_bound_counts["local_costs_with_non_zero_lower_and_finite_upper"]
        logger.info("Total number of groups with exact costs for all plans: %d" % len(exact_costs))
        meta_data["groups_with_exact_costs"] = len(exact_costs)
        logger.info("Groups in memory: %d" % len(self._memory))
        meta_data["groups_in_memory"] = len(self._memory)
        logger.info("Target encodings: %d" % len(self._target_encodings))
        meta_data["target_encodings"] = len(self._target_encodings)

        torch.cuda.empty_cache()

        def restore_module():
            if self._double_q_learning:
                self._module_a.load_state_dict(self._module_checkpoint_a)
                for param_group in self._optimizer_a.param_groups:
                    param_group['lr'] *= 0.5
                self._module_b.load_state_dict(self._module_checkpoint_b)
                for param_group in self._optimizer_b.param_groups:
                    param_group['lr'] *= 0.5
            else:
                self._module.load_state_dict(self._module_checkpoint)
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] *= 0.5

        batches = self._build_batches(exact_cost_target_indexes)
        logger.info("Batches per training epoch: %d" % len(batches))
        meta_data["batches_per_epoch"] = len(batches)
        mean_loss = None
        for epoch in range(int(self._current_num_training_epochs)):
            try:
                mean_loss = self._training_epoch(epoch, batches, logger)
                if mean_loss == 0:  # all batches skipped
                    restore_module()
                    return self._train(logger, lock)
            except ValueError:
                restore_module()
                return self._train(logger, lock)


        meta_data["final_mean_loss"] = mean_loss
        logger.info("")
        self._module_to(self._inference_device)
        self.trained = True

        if self._training_epoch_decay is not None:
            self._current_num_training_epochs = self._num_training_epochs + (self._current_num_training_epochs - self._num_training_epochs) * self._training_epoch_decay

        gc.collect()

        return meta_data

    def _training_epoch(self, epoch: int, batches: List[Tuple[List[TopDownLearnedOptimizerData], Union[TopDownLearnedOptimizerTargetCalculationData, List[TopDownLearnedOptimizerTargetData]]]], logger: logging.Logger = None, differentiate_double: bool = True) -> float:
        total_loss = 0
        for input_datas, target_calculation in batches:
            total_loss += self._training_batch(input_datas, target_calculation, differentiate_double, logger)
        mean_loss = total_loss / len(batches)
        logger.info("Epoch %d/%d: Loss: %.4f - %s" % (epoch + 1, self._current_num_training_epochs, mean_loss, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return mean_loss

    def _training_batch(self, input_datas: List[TopDownLearnedOptimizerData], target_calculation: Union[TopDownLearnedOptimizerTargetCalculationData, List[TopDownLearnedOptimizerTargetData]], differentiate_double: bool, logger: logging.Logger) -> float:
        def _check_model_corruption(module: torch.nn.Module) -> bool:
            for param in module.parameters():
                if not torch.isfinite(param).all():
                    return True
            return False

        total_loss = 0
        if self._double_q_learning:
            if isinstance(target_calculation, TopDownLearnedOptimizerTargetCalculationData):
                target_datas = self._calculate_targets_double(target_calculation)
                if not differentiate_double:
                    new_target_datas = []
                    for target_data_a, target_data_b in target_datas:
                        new_target_datas.append((target_data_a, target_data_a))
                    target_datas = new_target_datas
            else:
                if differentiate_double:
                    a_target_data, b_target_data = target_calculation
                    a_target_data = a_target_data.to(self._training_device)
                    b_target_data = b_target_data.to(self._training_device)
                else:
                    target_calculation = target_calculation[0].to(self._training_device)
                    a_target_data = target_calculation
                    b_target_data = target_calculation
                target_datas = [(a_target_data, b_target_data)]
            for input_data, (a_target_data, b_target_data) in zip(input_datas, target_datas):
                input_data = input_data.to(self._training_device)
                self._optimizer_a.zero_grad()
                loss_a, _, _ = self._module_a.loss(input_data, a_target_data)
                if torch.isfinite(loss_a):
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(self._module_a.parameters(), max_norm=1.0)
                    self._optimizer_a.step()
                    total_loss += loss_a.item() / 2
                else:
                    if _check_model_corruption(self._module_a):
                        logger.error("Model A parameters contain non-finite values. Stopping training.")
                        raise ValueError("Model A parameters contain non-finite values.")
                    logger.error("Optimizer A encountered a non-finite loss value.")

                self._optimizer_b.zero_grad()
                loss_b, _, _ = self._module_b.loss(input_data, b_target_data)
                if torch.isfinite(loss_b):
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(self._module_b.parameters(), max_norm=1.0)
                    self._optimizer_b.step()
                    total_loss += loss_b.item() / 2
                else:
                    if _check_model_corruption(self._module_b):
                        logger.error("Model B parameters contain non-finite values. Stopping training.")
                        raise ValueError("Model B parameters contain non-finite values.")
                    logger.error("Optimizer B encountered a non-finite loss value.")
        else:
            if isinstance(target_calculation, list):
                target_datas = [target_data.to(self._training_device) for target_data in target_calculation]
            else:
                target_datas = self._calculate_targets(target_calculation)
            for input_data, target_data in zip(input_datas, target_datas):
                input_data = input_data.to(self._training_device)
                self._optimizer.zero_grad()
                loss, _, _ = self._module.loss(input_data, target_data)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                    self._optimizer.step()
                    total_loss += loss.item()
                else:
                    if _check_model_corruption(self._module):
                        logger.error("Model parameters contain non-finite values. Stopping training.")
                        raise ValueError("Model parameters contain non-finite values.")
                    logger.error("Optimizer encountered a non-finite loss value.")
        torch.cuda.empty_cache()
        return total_loss

    def _num_batches(self, num_training_samples: int) -> int:
        return math.ceil(num_training_samples / self._max_batch_size)

    def _build_batches(self, exact_cost_target_indexes: Dict[int, float]) -> List[Tuple[List[TopDownLearnedOptimizerData], TopDownLearnedOptimizerTargetCalculationData]]:
        packs = {query: len(groups) for query, groups in self._memory_groups.items()}
        bins = binpacking.to_constant_volume(packs, self._max_batch_size)
        random.shuffle(bins)
        memory_batches = []
        for bin in bins:
            memory_batch = []
            for query in bin:
                memory_batch.extend(self._memory_groups[query])
            memory_batches.append([self._memory[group] for group in memory_batch])

        batches = []
        for memory_batch in memory_batches:
            if len(memory_batch) > self._max_batch_size:
                num_internal_batches = math.ceil(len(memory_batch) / self._max_batch_size)
                random.shuffle(memory_batch)
                internal_batches = np.array_split(memory_batch, num_internal_batches)
            else:
                internal_batches = [memory_batch]

            targets = []
            target_index = {}
            exact_costs = []
            exact_cost_index = {}

            for group_memory in memory_batch:
                for memoized_expression_memory in group_memory.memoized_expression_memories:
                    if memoized_expression_memory.target_memory_indexes is None:
                        target_memory_indexes = []
                    else:
                        target_memory_indexes = memoized_expression_memory.target_memory_indexes
                    for target_memory_index in target_memory_indexes + group_memory.subgroup_target_memory_indexes:
                        if target_memory_index in exact_cost_target_indexes:
                            if target_memory_index not in exact_cost_index:
                                exact_cost_index[target_memory_index] = len(exact_costs)
                                exact_costs.append(exact_cost_target_indexes[target_memory_index])
                        else:
                            if target_memory_index not in target_index:
                                target_index[target_memory_index] = len(targets)
                                targets.append(self._target_encodings[target_memory_index])

            if len(exact_costs) > 0:
                target_length = len(targets)
                for target_memory_index, index in exact_cost_index.items():
                    target_index[target_memory_index] = target_length + index

            input_datas = []
            cost_datas = []
            for internal_batch in internal_batches:
                data_batch = [query_memory.encoding for query_memory in internal_batch]
                input_loader = torch_geometric.loader.DataLoader(data_batch, batch_size=len(data_batch))
                input_data = next(iter(input_loader))
                input_datas.append(input_data)

                join_operator_outer_index = []
                join_operator_inner_index = []
                join_operator_subgroup_index = []

                join_operator_max_costs = []
                join_operator_min_costs = []
                join_operator_masks = []
                scan_operator_max_costs = []
                scan_operator_min_costs = []
                scan_operator_masks = []
                for group_memory in internal_batch:
                    plan_type_counts = {plan_type: 0 for plan_type in PlanType}
                    for memoized_expression_memory in group_memory.memoized_expression_memories:
                        plan_type_counts[memoized_expression_memory.plan_type] += 1

                    join_operator_min_cost = np.ones(plan_type_counts[PlanType.JOIN])
                    join_operator_max_cost = np.ones(plan_type_counts[PlanType.JOIN])
                    join_operator_mask = np.zeros(plan_type_counts[PlanType.JOIN], dtype=bool)
                    scan_operator_min_cost = np.ones(plan_type_counts[PlanType.SCAN])
                    scan_operator_max_cost = np.ones(plan_type_counts[PlanType.SCAN])
                    scan_operator_mask = np.zeros(plan_type_counts[PlanType.SCAN], dtype=bool)
                    for memoized_expression_memory in group_memory.memoized_expression_memories:
                        if memoized_expression_memory.max_local_cost is None:
                            max_base_cost = self._min_local_cost
                        else:
                            max_base_cost = max(memoized_expression_memory.max_local_cost, self._min_local_cost)
                        if memoized_expression_memory.min_local_cost is None:
                            min_base_cost = self._min_local_cost
                        else:
                            min_base_cost = max(memoized_expression_memory.min_local_cost, self._min_local_cost)
                        if memoized_expression_memory.plan_type == PlanType.JOIN:
                            join_operator_min_cost[memoized_expression_memory.plan_index] = min_base_cost
                            join_operator_max_cost[memoized_expression_memory.plan_index] = max_base_cost
                            if memoized_expression_memory.max_local_cost is not None:
                                join_operator_mask[memoized_expression_memory.plan_index] = True
                            if not self._learn_decomposed_costs:
                                if memoized_expression_memory.target_memory_indexes is None:
                                    join_operator_outer_index.append(-1)
                                    join_operator_inner_index.append(-1)
                                else:
                                    outer_target_index, inner_target_index = memoized_expression_memory.target_memory_indexes
                                    outer_index = target_index[outer_target_index]
                                    inner_index = target_index[inner_target_index]
                                    join_operator_outer_index.append(outer_index)
                                    join_operator_inner_index.append(inner_index)
                        elif memoized_expression_memory.plan_type == PlanType.SCAN:
                            scan_operator_min_cost[memoized_expression_memory.plan_index] = min_base_cost
                            scan_operator_max_cost[memoized_expression_memory.plan_index] = max_base_cost
                            if memoized_expression_memory.max_local_cost is not None:
                                scan_operator_mask[memoized_expression_memory.plan_index] = True
                        else:
                            raise NotImplementedError
                    if self._learn_decomposed_costs:
                        for subgroup_target_index in group_memory.subgroup_target_memory_indexes:
                            subgroup_index = target_index[subgroup_target_index]
                            join_operator_subgroup_index.append(subgroup_index)
                    join_operator_min_costs.append(torch.tensor(join_operator_min_cost, dtype=torch.float32))
                    join_operator_max_costs.append(torch.tensor(join_operator_max_cost, dtype=torch.float32))
                    join_operator_masks.append(torch.tensor(join_operator_mask, dtype=torch.bool))
                    scan_operator_min_costs.append(torch.tensor(scan_operator_min_cost, dtype=torch.float32))
                    scan_operator_max_costs.append(torch.tensor(scan_operator_max_cost, dtype=torch.float32))
                    scan_operator_masks.append(torch.tensor(scan_operator_mask, dtype=torch.bool))

                join_operator_outer_index = torch.tensor(join_operator_outer_index, dtype=torch.long)
                join_operator_inner_index = torch.tensor(join_operator_inner_index, dtype=torch.long)
                join_operator_subgroup_index = torch.tensor(join_operator_subgroup_index, dtype=torch.long)

                join_operator_max_costs = torch.cat(join_operator_max_costs, dim=0)
                join_operator_min_costs = torch.cat(join_operator_min_costs, dim=0)
                join_operator_masks = torch.cat(join_operator_masks, dim=0)
                scan_operator_max_costs = torch.cat(scan_operator_max_costs, dim=0)
                scan_operator_min_costs = torch.cat(scan_operator_min_costs, dim=0)
                scan_operator_masks = torch.cat(scan_operator_masks, dim=0)
                cost_data = TopDownLearnedOptimizerTargetCalculationCostData(join_operator_min_costs,
                                                                            join_operator_max_costs,
                                                                            join_operator_masks,
                                                                            join_operator_outer_index,
                                                                            join_operator_inner_index,
                                                                            join_operator_subgroup_index,
                                                                            scan_operator_min_costs,
                                                                            scan_operator_max_costs,
                                                                            scan_operator_masks)
                cost_datas.append(cost_data)

            target_batch_loader = torch_geometric.loader.DataLoader(targets, batch_size=self._target_batch_size)
            targets = []
            for target_batch in target_batch_loader:
                targets.append(target_batch)

            exact_costs = torch.tensor(exact_costs, dtype=torch.float32)
            target_data = TopDownLearnedOptimizerTargetCalculationData(targets,
                                                                       exact_costs,
                                                                       cost_datas)
            batches.append((input_datas, target_data))
        return batches

    def _calculate_targets(self, target_calculation_data: TopDownLearnedOptimizerTargetCalculationData) -> List[TopDownLearnedOptimizerTargetData]:
        target_calculation_data = target_calculation_data.to(self._training_device)
        target_datas = []
        with torch.no_grad():
            log_min_costs = []
            for target in target_calculation_data.targets:
                log_join_action_costs, log_scan_action_costs = self._module.log_forward(target)
                log_costs = torch.cat((log_join_action_costs, log_scan_action_costs), dim=0)
                cost_batches = torch.cat((target.join_operator_batch, target.scan_operator_batch), dim=0)

                log_min_cost = torch_scatter.scatter_min(log_costs, cost_batches, dim=0, dim_size=target.num_graphs)[0]
                log_min_costs.append(log_min_cost)

            log_min_costs.append(torch.log(target_calculation_data.fixed_subgroup_costs))
            log_target_costs = torch.cat(log_min_costs, dim=0)

            for cost_data in target_calculation_data.cost_datas:
                log_base_min = torch.log(cost_data.join_operator_min_cost)
                log_base_max = torch.log(cost_data.join_operator_max_cost)

                if self._learn_decomposed_costs:
                    log_subgroup_costs = torch.index_select(log_target_costs, 0, cost_data.join_operator_subgroup_index)
                    log_join_min_costs = log_base_min
                    log_join_max_costs = log_base_max
                else:
                    log_subgroup_costs = None
                    log_outer_join_costs = torch.index_select(log_target_costs, 0, cost_data.join_operator_outer_index)
                    log_inner_join_costs = torch.index_select(log_target_costs, 0, cost_data.join_operator_inner_index)

                    log_children = torch.logaddexp(log_outer_join_costs, log_inner_join_costs)
                    log_join_min_costs = torch.logaddexp(log_base_min, log_children)
                    log_join_max_costs = torch.logaddexp(log_base_max, log_children)

                target_data = TopDownLearnedOptimizerTargetData(log_join_min_costs,
                                                                log_join_max_costs,
                                                                cost_data.join_operator_cost_mask,
                                                                log_subgroup_costs,
                                                                torch.log(cost_data.scan_operator_min_cost),
                                                                torch.log(cost_data.scan_operator_max_cost),
                                                                cost_data.scan_operator_cost_mask)
                target_datas.append(target_data)
        return target_datas

    def _calculate_targets_double(self, target_calculation_data: TopDownLearnedOptimizerTargetCalculationData) -> List[
        Tuple[TopDownLearnedOptimizerTargetData, TopDownLearnedOptimizerTargetData]]:
        target_calculation_data = target_calculation_data.to(self._training_device)
        target_datas = []
        with torch.no_grad():
            min_costs_a = []
            min_costs_b = []
            for target in target_calculation_data.targets:
                log_join_a, log_scan_a = self._module_a.log_forward(target)
                log_join_b, log_scan_b = self._module_b.log_forward(target)

                log_costs_a = torch.cat((log_join_a, log_scan_a), dim=0)
                log_costs_b = torch.cat((log_join_b, log_scan_b), dim=0)
                cost_batches = torch.cat((target.join_operator_batch, target.scan_operator_batch), dim=0)

                indices_a = torch_scatter.scatter_min(log_costs_a, cost_batches, dim=0, dim_size=target.num_graphs)[1]
                indices_b = torch_scatter.scatter_min(log_costs_b, cost_batches, dim=0, dim_size=target.num_graphs)[1]

                log_min_cost_a = log_costs_b[indices_a]
                log_min_cost_b = log_costs_a[indices_b]

                min_costs_a.append(log_min_cost_a)
                min_costs_b.append(log_min_cost_b)

            min_costs_a.append(torch.log(target_calculation_data.fixed_subgroup_costs))
            min_costs_b.append(torch.log(target_calculation_data.fixed_subgroup_costs))

            target_costs_a = torch.cat(min_costs_a, dim=0)
            target_costs_b = torch.cat(min_costs_b, dim=0)

            for cost_data in target_calculation_data.cost_datas:
                log_base_min = torch.log(cost_data.join_operator_min_cost)
                log_base_max = torch.log(cost_data.join_operator_max_cost)

                if self._learn_decomposed_costs:
                    log_subgroup_costs_a = torch.index_select(target_costs_a, 0, cost_data.join_operator_subgroup_index)
                    log_subgroup_costs_b = torch.index_select(target_costs_b, 0, cost_data.join_operator_subgroup_index)
                    log_join_min_costs_a = log_base_min
                    log_join_max_costs_a = log_base_max
                    log_join_min_costs_b = log_base_min
                    log_join_max_costs_b = log_base_max
                else:
                    log_subgroup_costs_a = None
                    log_subgroup_costs_b = None

                    log_outer_join_costs_a = torch.index_select(target_costs_a, 0, cost_data.join_operator_outer_index)
                    log_inner_join_costs_a = torch.index_select(target_costs_a, 0, cost_data.join_operator_inner_index)
                    log_children_a = torch.logaddexp(log_outer_join_costs_a, log_inner_join_costs_a)
                    log_join_min_costs_a = torch.logaddexp(log_base_min, log_children_a)
                    log_join_max_costs_a = torch.logaddexp(log_base_max, log_children_a)

                    log_outer_join_costs_b = torch.index_select(target_costs_b, 0, cost_data.join_operator_outer_index)
                    log_inner_join_costs_b = torch.index_select(target_costs_b, 0, cost_data.join_operator_inner_index)
                    log_children_b = torch.logaddexp(log_outer_join_costs_b, log_inner_join_costs_b)
                    log_join_min_costs_b = torch.logaddexp(log_base_min, log_children_b)
                    log_join_max_costs_b = torch.logaddexp(log_base_max, log_children_b)

                log_scan_operator_min_cost = torch.log(cost_data.scan_operator_min_cost)
                log_scan_operator_max_cost = torch.log(cost_data.scan_operator_max_cost)

                target_data_a = TopDownLearnedOptimizerTargetData(log_join_min_costs_a,
                                                                  log_join_max_costs_a,
                                                                  cost_data.join_operator_cost_mask,
                                                                  log_subgroup_costs_a,
                                                                  log_scan_operator_min_cost,
                                                                  log_scan_operator_max_cost,
                                                                  cost_data.scan_operator_cost_mask)
                target_data_b = TopDownLearnedOptimizerTargetData(log_join_min_costs_b,
                                                                  log_join_max_costs_b,
                                                                  cost_data.join_operator_cost_mask,
                                                                  log_subgroup_costs_b,
                                                                  log_scan_operator_min_cost,
                                                                  log_scan_operator_max_cost,
                                                                  cost_data.scan_operator_cost_mask)

                target_datas.append((target_data_a, target_data_b))
        return target_datas

    def get_subgroup_costs_single(self, target_batches: List[TopDownLearnedOptimizerData]) -> Dict[int, List[float]]:
        subgroup_costs = {}
        group_counter = 0
        for target_batch in target_batches:
            with torch.no_grad():
                join_action_costs, scan_action_costs = self._module.forward(target_batch)
                costs = torch.cat((join_action_costs, scan_action_costs), dim=0)
                cost_batches = torch.cat((target_batch.join_operator_batch, target_batch.scan_operator_batch), dim=0)
                min_costs = torch_scatter.scatter_min(costs, cost_batches, dim=0, dim_size=target_batch.num_graphs)[0].detach().cpu().numpy()
            for min_cost in min_costs:
                subgroup_costs[group_counter] = [min_cost]
                group_counter += 1
        return subgroup_costs

    def get_subgroup_costs_double(self, target_batches: List[TopDownLearnedOptimizerData]) -> Dict[int, List[float]]:
        subgroup_costs = {}
        group_counter = 0
        for target_batch in target_batches:
            with torch.no_grad():
                join_action_costs_a, scan_action_costs_a = self._module_a.forward(target_batch)
                join_action_costs_b, scan_action_costs_b = self._module_b.forward(target_batch)
                costs_a = torch.cat((join_action_costs_a, scan_action_costs_a), dim=0)
                costs_b = torch.cat((join_action_costs_b, scan_action_costs_b), dim=0)
                cost_batches = torch.cat((target_batch.join_operator_batch, target_batch.scan_operator_batch), dim=0)
                min_costs_a = costs_b[torch_scatter.scatter_min(costs_a, cost_batches, dim=0, dim_size=target_batch.num_graphs)[1]].detach().cpu().numpy()
                min_costs_b = costs_a[torch_scatter.scatter_min(costs_b, cost_batches, dim=0, dim_size=target_batch.num_graphs)[1]].detach().cpu().numpy()
            for min_cost_a, min_cost_b in zip(min_costs_a, min_costs_b):
                subgroup_costs[group_counter] = [min_cost_a, min_cost_b]
                group_counter += 1
        return subgroup_costs

    def print_parameters(self, logger: logging.Logger):
        if self._double_q_learning:
            print_parameters(self._module_a, logger)
        else:
            print_parameters(self._module, logger)

    def save(self, path: str):
        if self._double_q_learning:
            module_path_a = path + "_module_a.pt"
            module_path_b = path + "_module_b.pt"
            optimizer_path_a = path + "_optimizer_a.pt"
            optimizer_path_b = path + "_optimizer_b.pt"
            torch.save(self._module_a.state_dict(), module_path_a)
            torch.save(self._module_b.state_dict(), module_path_b)
            torch.save(self._optimizer_a.state_dict(), optimizer_path_a)
            torch.save(self._optimizer_b.state_dict(), optimizer_path_b)
        else:
            module_path = path + "_module.pt"
            optimizer_path = path + "_optimizer.pt"
            torch.save(self._module.state_dict(), module_path)
            torch.save(self._optimizer.state_dict(), optimizer_path)
        self._encoder.save(path)

    def load(self, path: str):
        if self._double_q_learning:
            module_path_a = path + "_module_a.pt"
            module_path_b = path + "_module_b.pt"
            optimizer_path_a = path + "_optimizer_a.pt"
            optimizer_path_b = path + "_optimizer_b.pt"
            self._module_a.load_state_dict(torch.load(module_path_a, map_location=self._inference_device))
            self._module_b.load_state_dict(torch.load(module_path_b, map_location=self._inference_device))
            self._optimizer_a.load_state_dict(torch.load(optimizer_path_a, map_location=self._inference_device))
            self._optimizer_b.load_state_dict(torch.load(optimizer_path_b, map_location=self._inference_device))
        else:
            module_path = path + "_module.pt"
            optimizer_path = path + "_optimizer.pt"
            self._module.load_state_dict(torch.load(module_path, map_location=self._inference_device))
            self._optimizer.load_state_dict(torch.load(optimizer_path, map_location=self._inference_device))
        self._encoder.load(path)
        self.trained = True

    @staticmethod
    def _enumerate_single_table_queries(schema: Schema) -> List[SPJQuery]:
        queries = []
        for table in schema.tables():
            table_occurrence = TableOccurrence(table, table.name())
            table_occurrence.set_predicate(TruePredicate())
            queries.append(SPJQuery([table_occurrence], [], []))
        return queries

    @staticmethod
    def _enumerate_fk_pk_queries(schema: Schema) -> List[Tuple[SPJQuery, bool]]:
        """Returns list of (query, is_complete_fk) tuples.
        is_complete_fk is True when the join column is the entire FK (single-column FK),
        meaning the FK-PK cardinality guarantee holds exactly."""
        queries = []
        for fk in schema.foreign_keys():
            is_single_column_fk = len(fk.mapping()) == 1
            for fk_col, pk_col in fk.mapping().items():
                fk_name = fk.foreign_key_table().name()
                pk_name = fk.primary_key_table().name()
                fk_alias = fk_name if fk_name != pk_name else fk_name + "_fk"
                pk_alias = pk_name if fk_name != pk_name else pk_name + "_pk"
                fk_table_occurrence = TableOccurrence(fk.foreign_key_table(), fk_alias)
                fk_table_occurrence.set_predicate(TruePredicate())
                pk_table_occurrence = TableOccurrence(fk.primary_key_table(), pk_alias)
                pk_table_occurrence.set_predicate(TruePredicate())
                join = Join([(fk_table_occurrence, fk_col), (pk_table_occurrence, pk_col)])
                queries.append((SPJQuery([fk_table_occurrence, pk_table_occurrence], [join], []), is_single_column_fk))
        return queries

    @staticmethod
    def _enumerate_fk_fk_queries(schema: Schema) -> List[SPJQuery]:
        queries = []
        foreign_keys = list(schema.foreign_keys())
        for i, fk1 in enumerate(foreign_keys):
            for fk2 in foreign_keys[i:]:
                if fk1.primary_key_table() != fk2.primary_key_table():
                    continue
                pk_cols_1 = set(fk1.mapping().values())
                pk_cols_2 = set(fk2.mapping().values())
                shared_pk_cols = pk_cols_1 & pk_cols_2
                if not shared_pk_cols:
                    continue
                pk_to_fk1 = {pk_col: fk_col for fk_col, pk_col in fk1.mapping().items()}
                pk_to_fk2 = {pk_col: fk_col for fk_col, pk_col in fk2.mapping().items()}
                for pk_col in shared_pk_cols:
                    fk1_name = fk1.foreign_key_table().name()
                    fk2_name = fk2.foreign_key_table().name()
                    fk1_alias = fk1_name if fk1_name != fk2_name else fk1_name + "_1"
                    fk2_alias = fk2_name if fk1_name != fk2_name else fk2_name + "_2"
                    fk1_table_occurrence = TableOccurrence(fk1.foreign_key_table(), fk1_alias)
                    fk1_table_occurrence.set_predicate(TruePredicate())
                    fk2_table_occurrence = TableOccurrence(fk2.foreign_key_table(), fk2_alias)
                    fk2_table_occurrence.set_predicate(TruePredicate())
                    join = Join([(fk1_table_occurrence, pk_to_fk1[pk_col]),
                                 (fk2_table_occurrence, pk_to_fk2[pk_col])])
                    queries.append(SPJQuery([fk1_table_occurrence, fk2_table_occurrence], [join], []))
        return queries

    def _add_statistical_cardinality_bounds(self):
        single_table_queries = self._enumerate_single_table_queries(self._schema)
        fk_pk_queries = self._enumerate_fk_pk_queries(self._schema)
        fk_fk_queries = self._enumerate_fk_fk_queries(self._schema)

        # Single table queries: exact cardinality from table statistics
        for query in single_table_queries:
            table_occurrence = list(query.table_occurrences())[0]
            cardinality = table_occurrence.table().cardinality()
            self._training_queries.add(query)
            self._subquery_origins[query] = query
            self._update_cardinality_ranges(query, CardinalityRange.exact_cardinality_range(cardinality))

        # FK-PK joins: cardinality = |FK_table| - null_count for complete FKs,
        # lower bound only for partial (composite) FK columns.
        for query, is_complete_fk in fk_pk_queries:
            join = list(query.joins())[0]
            equivalence_class = list(join.equivalence_class())
            to_a, col_a = equivalence_class[0]
            to_b, col_b = equivalence_class[1]
            # Determine which side is the FK
            fk_to, fk_col = (to_a, col_a) if col_b.distinct_count() == to_b.table().cardinality() else (to_b, col_b)
            null_count = fk_col.null_count()
            if null_count is not None:
                lower_bound = fk_to.table().cardinality() - null_count
            else:
                lower_bound = 0
            if is_complete_fk:
                cardinality_range = CardinalityRange(lower_bound, lower_bound if null_count is not None else fk_to.table().cardinality())
            else:
                cardinality_range = CardinalityRange(lower_bound, None)
            self._training_queries.add(query)
            self._subquery_origins[query] = query
            self._update_cardinality_ranges(query, cardinality_range)

        # FK-FK joins: lower bound via MCV overlap
        for query in fk_fk_queries:
            join = list(query.joins())[0]
            equivalence_class = list(join.equivalence_class())
            to_a, col_a = equivalence_class[0]
            to_b, col_b = equivalence_class[1]
            card_a = to_a.table().cardinality()
            card_b = to_b.table().cardinality()
            is_self_join = to_a.table() == to_b.table() and col_a == col_b

            mcv_items_a = col_a.mcv_items()
            mcv_items_b = col_b.mcv_items()

            lower_bound = 0
            if mcv_items_a is not None and mcv_items_b is not None:
                # MCV overlap: use floor - 1 per count to absorb float rounding
                freq_dict_b = dict(mcv_items_b)
                for val, freq_a in mcv_items_a:
                    if val in freq_dict_b:
                        count_a = math.floor(card_a * freq_a) - 1
                        count_b = math.floor(card_b * freq_dict_b[val]) - 1
                        if count_a > 0 and count_b > 0:
                            lower_bound += count_a * count_b

                # Self-join: non-MCV values also contribute
                if is_self_join:
                    null_count = col_a.null_count()
                    mcv_freq_sum = sum(freq for _, freq in mcv_items_a)
                    distinct_count = col_a.distinct_count()
                    n_mcv = len(mcv_items_a)
                    if distinct_count is not None and distinct_count > n_mcv:
                        remaining_distinct = distinct_count - n_mcv
                        if null_count is not None:
                            remaining_rows = card_a - null_count - math.ceil(card_a * mcv_freq_sum) - 1
                            if remaining_rows > 0:
                                count_per_value = remaining_rows // remaining_distinct
                                if count_per_value > 0:
                                    lower_bound += remaining_distinct * count_per_value * count_per_value
                        else:
                            # Can't bound per-value counts, but each remaining
                            # distinct value generates at least 1*1 = 1 tuple.
                            lower_bound += remaining_distinct

            if lower_bound > 0:
                self._training_queries.add(query)
                self._subquery_origins[query] = query
                self._update_cardinality_ranges(query, CardinalityRange(lower_bound, None))
