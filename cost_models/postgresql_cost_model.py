import math
from typing import Dict, FrozenSet, Iterable, Optional, Set, Tuple, List

import numpy as np

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cost_models.cost_model import CostModel
from cost_models.local_cost_model import LocalCostModel
from queries.predicates.comparison_operator import COMPARISON_OPERATOR_EQ, COMPARISON_OPERATOR_IN, COMPARISON_OPERATOR_LIKE, COMPARISON_OPERATOR_ILIKE, COMPARISON_OPERATOR_LT, COMPARISON_OPERATOR_LTE, COMPARISON_OPERATOR_GT, COMPARISON_OPERATOR_GTE
from queries.predicates.comparison_predicate import ComparisonPredicate
from queries.predicates.conjunction import Conjunction
from queries.predicates.disjunction import Disjunction
from queries.predicates.join import Join
from queries.predicates.negation import Negation
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.predicates.true_predicate import TruePredicate
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.column import Column
from schemas.index import Index
from schemas.postgresql_configuration import PostgreSQLConfiguration
from schemas.schema import Schema
from schemas.table import Table


class PostgreSQLCostModel(LocalCostModel):
    def __init__(self,
                 cardinality_estimator: CardinalityEstimator,
                 postgresql_configuration: PostgreSQLConfiguration,
                 schema: Optional[Schema] = None,
                 default_width: Optional[int] = None,
                 improved: bool = False,
                 index_scan_base_cost: float = 1,
                 probe_cache_penalty: float = 1.0):
        super().__init__(cardinality_estimator)
        self._postgresql_configuration = postgresql_configuration
        self._schema = schema
        self._default_width = default_width
        self._improved = improved
        self._index_scan_base_cost = index_scan_base_cost
        # Multiplier applied to per-tuple CPU costs at cache-sensitive sites (hash probe/build, NL qual loop, Material rescan, MergeJoin inner materialize). Compensates for PG's flat cpu_operator_cost not modeling DRAM/LLC-miss latency when in-memory structures exceed CPU cache.
        self._probe_cache_penalty = probe_cache_penalty
        # Lazy cache of max_deg_T(S) = MAX(COUNT(*)) GROUP BY S over table T (NULLs excluded).
        # Queried on demand from bound propagation; shared across replace_cardinality_estimator clones.
        self._composite_max_degrees: Dict[Tuple[Table, FrozenSet[Column]], int] = {}
        self.debug_hash_join = False
        self.debug_nested_loop = False
        self.debug_merge_join = False

    def replace_cardinality_estimator(self, cardinality_estimator: CardinalityEstimator) -> CostModel:
        clone = PostgreSQLCostModel(cardinality_estimator, self._postgresql_configuration, schema=self._schema, default_width=self._default_width, improved=self._improved, index_scan_base_cost=self._index_scan_base_cost, probe_cache_penalty=self._probe_cache_penalty)
        clone._composite_max_degrees = self._composite_max_degrees
        return clone

    def composite_min_degree_lower_bound(self, table: Table, columns: Iterable[Column]) -> Optional[int]:
        # Per-key lower bound on inner rows when matched_prefix_columns are bound by an
        # outer join condition. Gated on full FK coverage: every value the outer can
        # bind must exist in `table` for `column.min_degree()` to be a valid floor.
        # Returns None when no safe bound is certifiable; 0 when the column is
        # FK-referencing but at least one parent has uncovered values.
        # Composite (multi-column) prefixes are deferred — return None.
        columns_frozen = frozenset(columns)
        if len(columns_frozen) != 1:
            return None
        (column,) = tuple(columns_frozen)
        min_deg = column.min_degree()
        if min_deg is None:
            return None
        if self._schema is None:
            return None
        fk_distinct = column.distinct_count()
        if fk_distinct is None:
            return None
        has_fk = False
        for fk in self._schema.foreign_keys_from(table):
            for fk_col, pk_col in fk.mapping().items():
                if fk_col != column:
                    continue
                has_fk = True
                pk_distinct = pk_col.distinct_count()
                if pk_distinct is None:
                    return None
                if fk_distinct < pk_distinct:
                    return 0
        if not has_fk:
            return None
        return int(min_deg)

    def composite_max_degree(self, table: Table, columns: Iterable[Column]) -> Optional[int]:
        # max_deg_T(S): upper bound on the number of T-rows sharing a given tuple of values on S,
        # with NULLs excluded (they never match in equi-joins). Returns None if S is empty or the
        # computed value is 0 (empty group relation — no equi-join matches possible). Cached per
        # (table, frozenset(columns)).
        columns_frozen = frozenset(columns)
        if not columns_frozen:
            return None
        key = (table, columns_frozen)
        if key in self._composite_max_degrees:
            cached = self._composite_max_degrees[key]
            return cached if cached > 0 else None
        if len(columns_frozen) == 1:
            (only_col,) = tuple(columns_frozen)
            precomputed = only_col.max_degree()
            if precomputed is not None:
                self._composite_max_degrees[key] = precomputed
                return precomputed if precomputed > 0 else None
        assert self._schema is not None, "PostgreSQLCostModel needs schema for composite_max_degree lookup"
        col_list = sorted(columns_frozen, key=lambda c: c.name())
        col_names = ", ".join(c.name() for c in col_list)
        not_null = " AND ".join("%s IS NOT NULL" % c.name() for c in col_list)
        query = "SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM %s WHERE %s GROUP BY %s) AS sub;" % (table.name(), not_null, col_names)
        cursor = self._schema.connection().cursor()
        cursor.execute(query)
        result = cursor.fetchone()[0]
        cursor.close()
        result = 0 if result is None else int(result)
        self._composite_max_degrees[key] = result
        return result if result > 0 else None

    def local_cost(self, relational_algebra_expression: RelationalAlgebraExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        if isinstance(relational_algebra_expression, HashJoinExpression):
            return self._hash_join_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        elif isinstance(relational_algebra_expression, MergeJoinExpression):
            return self._merge_join_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        elif isinstance(relational_algebra_expression, NestedLoopJoinExpression):
            return self._nested_loop_join_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        elif isinstance(relational_algebra_expression, SequentialScanExpression):
            return self._sequential_scan_cost(relational_algebra_expression)
        elif isinstance(relational_algebra_expression, IndexScanExpression):
            if relational_algebra_expression.join_clause:
                return self._index_scan_base_cost
            else:
                return self._where_clause_index_scan_cost(relational_algebra_expression, cardinality_mode=cardinality_mode)
        else:
            raise NotImplementedError

    def _predicate_cost_per_tuple(self, predicate: Predicate) -> float:
        if isinstance(predicate, TruePredicate):
            return 0.0
        elif isinstance(predicate, Conjunction):
            return sum(self._predicate_cost_per_tuple(p) for p in predicate.predicates())
        elif isinstance(predicate, Disjunction):
            return sum(self._predicate_cost_per_tuple(p) for p in predicate.predicates())
        elif isinstance(predicate, Negation):
            return self._predicate_cost_per_tuple(predicate.predicate()) + self._postgresql_configuration.cpu_operator_cost
        elif isinstance(predicate, SimplePredicate):
            if predicate.operator() == COMPARISON_OPERATOR_IN:
                value = predicate.value().value()
                assert isinstance(value, tuple)
                array_length = len(value)
                if array_length > 8:
                    array_length = 4
                return max(1, array_length / 2) * self._postgresql_configuration.cpu_operator_cost
            elif predicate.operator() in (COMPARISON_OPERATOR_LIKE, COMPARISON_OPERATOR_ILIKE) and self._improved:
                return 3 * self._postgresql_configuration.cpu_operator_cost
            else:
                return self._postgresql_configuration.cpu_operator_cost
        elif isinstance(predicate, Join):
            return self._postgresql_configuration.cpu_operator_cost
        elif isinstance(predicate, ComparisonPredicate):
            return self._postgresql_configuration.cpu_operator_cost
        else:
            raise NotImplementedError

    def cpu_operator_cost_per_tuple(self, table_occurrence: TableOccurrence) -> float:
        predicate = table_occurrence.predicate()
        return self._predicate_cost_per_tuple(predicate) / self._postgresql_configuration.cpu_operator_cost

    def _strip_index_covered_predicates(self, predicate: Predicate, index_columns: Set[Column]) -> Predicate:
        """Return predicate with equality conditions on index_columns removed."""
        if isinstance(predicate, TruePredicate):
            return predicate
        elif isinstance(predicate, SimplePredicate):
            if predicate.operator() == COMPARISON_OPERATOR_EQ and predicate.column() in index_columns:
                return TruePredicate()
            return predicate
        elif isinstance(predicate, Conjunction):
            remaining = [p for p in predicate.predicates()
                         if not (isinstance(p, SimplePredicate)
                                 and p.operator() == COMPARISON_OPERATOR_EQ
                                 and p.column() in index_columns)]
            if len(remaining) == 0:
                return TruePredicate()
            elif len(remaining) == 1:
                return remaining[0]
            else:
                return Conjunction(remaining)
        else:
            return predicate

    def _num_join_conditions(self, join_condition: Join) -> int:
        if isinstance(join_condition, TruePredicate):
            return 0
        elif isinstance(join_condition, Conjunction):
            return len(join_condition.predicates())
        elif isinstance(join_condition, Join):
            return len(join_condition.equivalence_class()) - 1
        else:
            raise ValueError("Unsupported join condition type: %s" % type(join_condition))

    def _hash_join_cost(self, hash_join_expression: HashJoinExpression, cardinality_mode: CardinalityMode) -> Optional[float]:
        join_cardinality_estimate = self.cardinality_estimator.estimate(hash_join_expression.canonical_query(), cardinality_mode=cardinality_mode)
        if join_cardinality_estimate is None:
            return None
        outer_cardinality_estimate = self.cardinality_estimator.estimate(hash_join_expression.outer.canonical_query(), cardinality_mode=cardinality_mode)
        if outer_cardinality_estimate is None:
            return None
        inner_cardinality_estimate = self.cardinality_estimator.estimate(hash_join_expression.inner.canonical_query(), cardinality_mode=cardinality_mode)
        if inner_cardinality_estimate is None:
            return None
        if isinstance(hash_join_expression.join_condition, Join):
            num_hash_clauses = 1
        elif isinstance(hash_join_expression.join_condition, Conjunction):
            num_hash_clauses = len(hash_join_expression.join_condition.predicates())
        else:
            raise ValueError("Unsupported join condition type: %s" % type(hash_join_expression.join_condition))
        hash_inner = inner_cardinality_estimate * (self._postgresql_configuration.cpu_tuple_cost + self._probe_cache_penalty * self._postgresql_configuration.cpu_operator_cost * num_hash_clauses)
        hash_outer = outer_cardinality_estimate * self._probe_cache_penalty * self._postgresql_configuration.cpu_operator_cost * num_hash_clauses
        inner_query = hash_join_expression.inner.query()
        inner_table_occurrences = inner_query.table_occurrences()
        if self._default_width is None:
            inner_tuple_width = inner_query.width()
        else:
            inner_tuple_width = self._default_width
        number_of_buckets, number_of_batches = self._choose_hash_table_size(inner_cardinality_estimate, inner_tuple_width)  # ExecChooseHashTableSize in PostgreSQL
        if number_of_batches > 1:
            if self._default_width is None:
                outer_tuple_width = hash_join_expression.outer.query().width()
            else:
                outer_tuple_width = self._default_width
            outer_pages = self._page_size(outer_cardinality_estimate, outer_tuple_width)
            inner_pages = self._page_size(inner_cardinality_estimate, inner_tuple_width)
            batching_cost = 2 * (outer_pages + inner_pages) * self._postgresql_configuration.seq_page_cost
        else:
            batching_cost = 0.0
        virtual_buckets = number_of_buckets * number_of_batches
        relation_was_made_unique = False  # TODO
        if relation_was_made_unique:
            inner_bucket_size = 1 / virtual_buckets
            inner_mcv_frequency = 0.0
        else:
            other_clauses = hash_join_expression.join_condition
            if isinstance(other_clauses, Join):
                other_clauses = [other_clauses]
            elif isinstance(other_clauses, Conjunction):
                other_clauses = other_clauses.predicates()
            else:
                raise ValueError("Unsupported join condition type: %s" % type(other_clauses))

            pk_covered = self._pk_covered_tables(other_clauses, inner_table_occurrences)

            # For each equivalence class, aggregate inner-side stats:
            #   - min(ndistinct) across inner columns (upper bound on join ndistinct)
            #   - max(mcv) across inner columns (joins amplify skew)
            # Then prune classes whose inner columns are all functionally
            # determined by PK columns covered by other classes.
            per_class_stats = []
            for join_clause in other_clauses:
                assert isinstance(join_clause, Join)
                class_ndistinct = None
                class_mcv = 0.0
                has_inner_column = False
                any_survived = False
                for table_occurrence, column in join_clause.equivalence_class():
                    if table_occurrence not in inner_table_occurrences:
                        continue
                    has_inner_column = True
                    # FD pruning: skip non-PK columns of tables whose PK is
                    # fully covered — they add no independent selectivity.
                    table = table_occurrence.table()
                    if self._is_fd_determined(table, column, pk_covered):
                        if self.debug_hash_join:
                            print(f"      [HashJoin FD prune] {table.name()}.{column.name()}")
                        continue
                    any_survived = True
                    adj_nd, mcv = self._adjusted_column_ndistinct_mcv(table_occurrence, column)
                    if adj_nd is not None and (class_ndistinct is None or adj_nd < class_ndistinct):
                        class_ndistinct = adj_nd
                    if mcv > class_mcv:
                        class_mcv = mcv
                assert has_inner_column
                if not any_survived:
                    continue

                per_class_stats.append((class_ndistinct, class_mcv))
                if self.debug_hash_join:
                    nd_str = f"{class_ndistinct:.1f}" if class_ndistinct is not None else "None"
                    print(f"      [HashJoin class] ndistinct={nd_str} mcv={class_mcv:.3e}")

            # Combine ndistinct and MCV across equivalence classes.
            # Interpolate (geometric mean) between the correlated and independent
            # extremes, with products bounded by physical limits.
            ndistincts = []
            mcvs = []
            for nd, mcv in per_class_stats:
                if nd is None:
                    nd = max(1.0, 1.0 / max(0.1, mcv))
                nd = max(nd, 1)
                effective_mcv = mcv if mcv > 0 else 1.0 / nd
                ndistincts.append(nd)
                mcvs.append(effective_mcv)

            inner_card_bound = max(inner_cardinality_estimate, 1.0)
            ndistinct_product = min(math.prod(ndistincts), inner_card_bound)
            combined_ndistinct = math.sqrt(max(ndistincts) * ndistinct_product)

            mcv_floor = 1.0 / inner_card_bound
            mcv_product = max(math.prod(mcvs), mcv_floor)
            combined_mcv = math.sqrt(min(mcvs) * mcv_product)

            # Compute bucket_size from combined stats.
            if combined_ndistinct > virtual_buckets:
                inner_bucket_size = 1.0 / virtual_buckets
            else:
                inner_bucket_size = 1.0 / combined_ndistinct
            average_frequency = 1.0 / combined_ndistinct
            if combined_mcv > average_frequency:
                inner_bucket_size *= combined_mcv / average_frequency
            inner_bucket_size = max(1e-6, min(inner_bucket_size, 1.0))
            inner_mcv_frequency = combined_mcv

            if self.debug_hash_join:
                print(
                    f"      [HashJoin combined] ndistinct={combined_ndistinct:.1f} mcv={combined_mcv:.3e} "
                    f"bucket_size={inner_bucket_size:.3e}"
                )

        spill_costs = 0
        mcv_count = self._clamp_row_est(inner_cardinality_estimate * inner_mcv_frequency)
        mcv_bucket_size = self._relation_byte_size(mcv_count, inner_tuple_width)
        hash_memory_limit = self._get_hash_memory_limit()
        if mcv_bucket_size > hash_memory_limit:
            overflow_bytes = mcv_bucket_size - hash_memory_limit
            overflow_pages = math.ceil(overflow_bytes / self._postgresql_configuration.block_size)
            spill_costs += 2 * overflow_pages * self._postgresql_configuration.seq_page_cost

        inner_unique = self._is_inner_unique(inner_query, hash_join_expression.join_condition, inner_table_occurrences)
        qual_cost_per_tuple = self._predicate_cost_per_tuple(hash_join_expression.join_condition)
        if inner_unique:
            # PG's final_cost_hashjoin inner_unique branch: matched outer rows pay
            # on average half the bucket scan (early-out once a match is found);
            # unmatched outer rows pay a small scan proportional to the whole
            # hash table's load factor to detect non-membership.
            outer_matched_rows = min(join_cardinality_estimate, outer_cardinality_estimate)
            outer_unmatched_rows = outer_cardinality_estimate - outer_matched_rows
            hash_eval = outer_matched_rows * self._clamp_row_est(inner_cardinality_estimate * inner_bucket_size) * qual_cost_per_tuple * 0.5 + qual_cost_per_tuple * outer_unmatched_rows * self._clamp_row_est(inner_cardinality_estimate / virtual_buckets) * 0.05
        else:
            hash_eval = outer_cardinality_estimate * self._clamp_row_est(inner_bucket_size * inner_cardinality_estimate) * qual_cost_per_tuple * 0.5  # cost_qual_eval in PostgreSQL
        post_join_cost = join_cardinality_estimate * self._postgresql_configuration.cpu_tuple_cost
        total = hash_inner + batching_cost + hash_outer + hash_eval + post_join_cost + spill_costs
        if self.debug_hash_join:
            print(
                f"    [HashJoin debug] outer={outer_cardinality_estimate:.0f} inner={inner_cardinality_estimate:.0f} join={join_cardinality_estimate:.0f} "
                f"clauses={num_hash_clauses} nbuckets={number_of_buckets} nbatches={number_of_batches} "
                f"bucket_size={inner_bucket_size:.3e} mcv_freq={inner_mcv_frequency:.3e} "
                f"inner_unique={inner_unique} "
                f"hash_inner={hash_inner:.2f} hash_outer={hash_outer:.2f} batching={batching_cost:.2f} "
                f"hash_eval={hash_eval:.2f} post_join={post_join_cost:.2f} spill={spill_costs:.2f} total={total:.2f}"
            )
        return total

    def _choose_hash_table_size(self, cardinality_estimate: float, tuple_width: int) -> Tuple[int, int]:
        hash_join_tuple_overhead = 16  # #define HJTUPLE_OVERHEAD  MAXALIGN(sizeof(HashJoinTupleData))
        size_of_minimal_tuple_header = 13  # #define SizeofMinimalTupleHeader offsetof(MinimalTupleData, t_bits)
        tuple_size = hash_join_tuple_overhead + self._max_align(size_of_minimal_tuple_header) + self._max_align(tuple_width)
        inner_relation_bytes = cardinality_estimate * tuple_size
        hash_table_bytes = self._get_hash_memory_limit()  # get_hash_memory_limit in PostgreSQL
        size_of_hash_join_tuple = 8
        max_alloc_size = 2 ** 30 - 1
        max_pointers = min(hash_table_bytes, max_alloc_size) // size_of_hash_join_tuple
        max_pointers = self._previous_power(max_pointers)  # pg_prevpower2_size_t(max_pointers)
        int_max = 2 ** 31 - 1
        max_pointers = min(max_pointers, int_max // 2 + 1)
        tuples_per_bucket = 1
        dbuckets = math.ceil(cardinality_estimate / tuples_per_bucket)
        dbuckets = min(dbuckets, max_pointers)
        nbuckets = max(dbuckets, 1024)
        nbuckets = self._next_power(nbuckets)  # pg_nextpower2_32(nbuckets)

        bucket_bytes = nbuckets * size_of_hash_join_tuple
        if inner_relation_bytes + bucket_bytes > hash_table_bytes:
            bucket_size = tuple_size * tuples_per_bucket + hash_join_tuple_overhead
            if hash_table_bytes <= bucket_size:
                s_buckets = 1
            else:
                s_buckets = self._next_power(hash_table_bytes // bucket_size)
            sbuckets = min(s_buckets, max_pointers)
            nbuckets = self._next_power(sbuckets)
            bucket_bytes = nbuckets * size_of_hash_join_tuple
            assert bucket_bytes <= hash_table_bytes // 2
            dbatch = math.ceil(inner_relation_bytes / (hash_table_bytes - bucket_bytes))
            dbatch = min(dbatch, max_pointers)
            nbatch = self._next_power(max(2, dbatch))
        else:
            nbatch = 1

        while nbatch > 0:
            current_space = hash_table_bytes + 2 * nbatch * self._postgresql_configuration.block_size
            new_space = hash_table_bytes * 2 + nbatch * self._postgresql_configuration.block_size
            if current_space < new_space:
                break
            nbatch = nbatch // 2
            nbuckets *= 2
        return nbuckets, nbatch

    def _get_hash_memory_limit(self) -> int:
        memory_limit = int(self._postgresql_configuration.work_mem * self._postgresql_configuration.hash_mem_multiplier)
        size_max = 2 ** 64 - 1
        memory_limit = min(memory_limit, size_max)
        return memory_limit

    @staticmethod
    def _previous_power(number: int) -> int:
        if number < 1:
            raise ValueError("Number must be greater than 0")
        power = 1
        while power * 2 < number:
            power *= 2
        return power

    @staticmethod
    def _next_power(number: int) -> int:
        if number < 1:
            raise ValueError("Number must be greater than 0")
        power = 1
        while power < number:
            power *= 2
        return power

    @staticmethod
    def _max_align(number: int, align: int = 8) -> int:
        return ((number + align - 1) // align) * align

    def _page_size(self, cardinality_estimate: float, width: int) -> int:
        size_of_heap_tuple_header = PostgreSQLCostModel._size_of_heap_tuple_header()
        relation_byte_size = cardinality_estimate * (self._max_align(size_of_heap_tuple_header) + self._max_align(width))
        return math.ceil(relation_byte_size / self._postgresql_configuration.block_size)

    @staticmethod
    def _clamp_row_est(row_est: float) -> float:
        maximum_row_count = 1e100
        if row_est < 1:
            return 1
        elif row_est > maximum_row_count:
            return maximum_row_count
        else:
            return row_est

    def _estimate_hash_bucket_stats(self, table_occurrence: TableOccurrence, column: Column, bucket_count: float) -> Tuple[float, float]:
        mcv_frequency = column.mcv_frequency()
        if mcv_frequency is None:
            mcv_frequency = 0.0

        distinct_count = column.distinct_count()
        if distinct_count is None:
            bucket_size_fraction = max(0.1, mcv_frequency)
            return mcv_frequency, bucket_size_fraction

        null_fraction = column.null_fraction()
        if null_fraction is None:
            null_fraction = 0.0

        average_frequency = (1 - null_fraction) / distinct_count

        if not isinstance(table_occurrence.predicate(), TruePredicate):
            table_occurrence_query = SPJQuery([table_occurrence], [], [])
            cardinality_estimate = self.cardinality_estimator.estimate(table_occurrence_query)
            if cardinality_estimate is not None:
                table_cardinality = table_occurrence.table().cardinality()
                if cardinality_estimate < table_cardinality:
                    distinct_count = distinct_count * cardinality_estimate / table_cardinality
            else:
                distinct_count = distinct_count * 0.1  # TODO
            if distinct_count < 1:
                distinct_count = 1

        if distinct_count > bucket_count:
            estimated_fraction = 1 / bucket_count
        else:
            estimated_fraction = 1 / distinct_count

        if average_frequency > 0 and mcv_frequency > average_frequency:
            estimated_fraction = estimated_fraction * mcv_frequency / average_frequency

        minimum_estimated_fraction = 1e-6
        if estimated_fraction < minimum_estimated_fraction:
            estimated_fraction = minimum_estimated_fraction
        elif estimated_fraction > 1.0:
            estimated_fraction = 1.0

        return mcv_frequency, estimated_fraction

    def _adjusted_column_ndistinct_mcv(self, table_occurrence: TableOccurrence, column: Column) -> Tuple[Optional[float], float]:
        """Return (predicate-adjusted ndistinct, mcv_frequency) for a column.
        ndistinct is None when column statistics are unavailable."""
        mcv_frequency = column.mcv_frequency()
        if mcv_frequency is None:
            mcv_frequency = 0.0
        distinct_count = column.distinct_count()
        if distinct_count is None:
            return None, mcv_frequency
        if not isinstance(table_occurrence.predicate(), TruePredicate):
            table_occurrence_query = SPJQuery([table_occurrence], [], [])
            cardinality_estimate = self.cardinality_estimator.estimate(table_occurrence_query)
            if cardinality_estimate is not None:
                table_cardinality = table_occurrence.table().cardinality()
                if cardinality_estimate < table_cardinality:
                    distinct_count = distinct_count * cardinality_estimate / table_cardinality
            else:
                distinct_count = distinct_count * 0.1
            if distinct_count < 1:
                distinct_count = 1
        return float(distinct_count), mcv_frequency

    @staticmethod
    def _pk_covered_tables(join_clauses, inner_table_occurrences):
        """Return the set of inner tables whose PK columns are all present
        as inner-side columns across the given join equivalence classes."""
        inner_columns_by_table = {}
        for join_clause in join_clauses:
            for table_occurrence, column in join_clause.equivalence_class():
                if table_occurrence in inner_table_occurrences:
                    table = table_occurrence.table()
                    if table not in inner_columns_by_table:
                        inner_columns_by_table[table] = set()
                    inner_columns_by_table[table].add(column)
        result = set()
        for table, columns in inner_columns_by_table.items():
            pk_columns = set(table.primary_key().columns())
            if pk_columns.issubset(columns):
                result.add(table)
        return result

    @staticmethod
    def _is_fd_determined(table, column, pk_covered_tables):
        """Return True if column is a non-PK column of a PK-covered table."""
        return table in pk_covered_tables and column not in set(table.primary_key().columns())

    @staticmethod
    def _relation_byte_size(cardinality_estimate: float, width: int) -> int:
        size_of_heap_tuple_header = PostgreSQLCostModel._size_of_heap_tuple_header()
        return math.ceil(cardinality_estimate * (PostgreSQLCostModel._max_align(size_of_heap_tuple_header) + PostgreSQLCostModel._max_align(width)))

    @staticmethod
    def _size_of_heap_tuple_header() -> int:
        return 17

    def _merge_join_cost(self, merge_join_expression: MergeJoinExpression, cardinality_mode: CardinalityMode) -> Optional[float]:
        outer_cardinality_estimate = self.cardinality_estimator.estimate(merge_join_expression.outer.canonical_query(), cardinality_mode=cardinality_mode)
        if outer_cardinality_estimate is None:
            return None
        inner_cardinality_estimate = self.cardinality_estimator.estimate(merge_join_expression.inner.canonical_query(), cardinality_mode=cardinality_mode)
        if inner_cardinality_estimate is None:
            return None
        join_cardinality_estimate = self.cardinality_estimator.estimate(merge_join_expression.canonical_query(), cardinality_mode=cardinality_mode)
        if join_cardinality_estimate is None:
            return None
        outer_cardinality_estimate = max(outer_cardinality_estimate, 1.0)
        inner_cardinality_estimate = max(inner_cardinality_estimate, 1.0)
        join_cardinality_estimate = max(join_cardinality_estimate, 1.0)

        outer_query = merge_join_expression.outer.query()
        inner_query = merge_join_expression.inner.query()
        assert isinstance(inner_query, SPJQuery)
        if self._default_width is None:
            outer_width = outer_query.width()
            inner_width = inner_query.width()
        else:
            outer_width = self._default_width
            inner_width = self._default_width

        outer_sort_startup, outer_sort_run = self._cost_sort(outer_cardinality_estimate, outer_width)
        inner_sort_startup, inner_sort_run = self._cost_sort(inner_cardinality_estimate, inner_width)

        startup_cost = outer_sort_startup + inner_sort_startup
        run_cost = outer_sort_run

        inner_table_occurrences = inner_query.table_occurrences()
        inner_unique = self._is_inner_unique(inner_query, merge_join_expression.join_condition, inner_table_occurrences)

        skip_mark_restore = inner_unique

        if skip_mark_restore:
            rescanned_tuples = 0.0
        else:
            rescanned_tuples = max(0.0, join_cardinality_estimate - inner_cardinality_estimate)
        rescan_ratio = 1.0 + rescanned_tuples / inner_cardinality_estimate

        bare_inner_cost = inner_sort_run * rescan_ratio
        mat_inner_cost = inner_sort_run + self._probe_cache_penalty * self._postgresql_configuration.cpu_operator_cost * inner_cardinality_estimate * rescan_ratio
        materialize_inner = (not skip_mark_restore) and mat_inner_cost < bare_inner_cost
        run_cost += mat_inner_cost if materialize_inner else bare_inner_cost

        merge_qual_per_tuple = self._predicate_cost_per_tuple(merge_join_expression.join_condition)
        run_cost += merge_qual_per_tuple * (outer_cardinality_estimate + inner_cardinality_estimate * rescan_ratio)
        run_cost += self._postgresql_configuration.cpu_tuple_cost * join_cardinality_estimate

        total = startup_cost + run_cost

        if self.debug_merge_join:
            print(
                f"    [MergeJoin debug] outer={outer_cardinality_estimate:.0f} inner={inner_cardinality_estimate:.0f} join={join_cardinality_estimate:.0f} "
                f"outer_sort=({outer_sort_startup:.2f}+{outer_sort_run:.2f}) inner_sort=({inner_sort_startup:.2f}+{inner_sort_run:.2f}) "
                f"inner_unique={inner_unique} rescan_ratio={rescan_ratio:.3f} materialize={materialize_inner} "
                f"bare_inner={bare_inner_cost:.2f} mat_inner={mat_inner_cost:.2f} total={total:.2f}"
            )

        return total

    def _cost_sort(self, tuples: float, width: int) -> Tuple[float, float]:
        """Reimplementation of PostgreSQL cost_tuplesort (comparison_cost=0, no LIMIT).
        Returns (startup_cost, run_cost) — does not include the child input cost."""
        if tuples < 2.0:
            tuples = 2.0
        comparison_cost = 2.0 * self._postgresql_configuration.cpu_operator_cost
        input_bytes = self._relation_byte_size(tuples, width)
        sort_mem_bytes = self._postgresql_configuration.work_mem * 1024
        if input_bytes > sort_mem_bytes:
            npages = math.ceil(input_bytes / self._postgresql_configuration.block_size)
            nruns = input_bytes / sort_mem_bytes
            merge_buffer_size = 65536
            minimal_merge_allowance = 1024
            min_order = 6
            max_order = 500
            mergeorder = (sort_mem_bytes - minimal_merge_allowance) / merge_buffer_size
            mergeorder = max(min_order, min(int(mergeorder), max_order))
            startup_cost = comparison_cost * tuples * math.log2(tuples)
            if nruns > mergeorder:
                log_runs = math.ceil(math.log(nruns) / math.log(mergeorder))
            else:
                log_runs = 1.0
            npageaccesses = 2.0 * npages * log_runs
            startup_cost += npageaccesses * (self._postgresql_configuration.seq_page_cost * 0.75 + self._postgresql_configuration.random_page_cost * 0.25)
        else:
            startup_cost = comparison_cost * tuples * math.log2(tuples)
        run_cost = self._postgresql_configuration.cpu_operator_cost * tuples
        return startup_cost, run_cost

    def _nested_loop_join_cost(self, nested_loop_join_expression: NestedLoopJoinExpression, cardinality_mode: CardinalityMode) -> Optional[float]:
        outer_cardinality_estimate = self.cardinality_estimator.estimate(nested_loop_join_expression.outer.canonical_query(), cardinality_mode=cardinality_mode)
        if outer_cardinality_estimate is None:
            return None
        outer_cardinality_estimate = max(outer_cardinality_estimate, 1.0)

        cost = 0

        if isinstance(nested_loop_join_expression.join_condition, Join):
            join_conditions = [nested_loop_join_expression.join_condition]
        elif isinstance(nested_loop_join_expression.join_condition, Conjunction):
            join_conditions = nested_loop_join_expression.join_condition.predicates()
        else:
            raise ValueError("Unsupported join condition type: %s" % type(nested_loop_join_expression.join_condition))

        outer_query = nested_loop_join_expression.outer.query()
        outer_table_occurrences = outer_query.table_occurrences()
        inner_table_occurrences_for_loop = nested_loop_join_expression.inner.query().table_occurrences()
        # Group loop counts by inner column to avoid overcounting when multiple
        # join conditions reference the same inner column. This arises in star
        # schemas where a multi-way equijoin (e.g., site_id joining all tables)
        # decomposes into one binary condition per outer table, all with the same
        # inner column. Taking the naive product gives N_sites^K instead of N_sites.
        inner_column_loop_counts = {}
        for join_condition in join_conditions:
            min_outer_loop = None
            for table_occurrence, column in join_condition.equivalence_class():
                if table_occurrence in outer_table_occurrences:
                    loop = self._estimate_num_groups(table_occurrence, column, outer_cardinality_estimate, cardinality_mode)
                    if loop is not None and (min_outer_loop is None or loop < min_outer_loop):
                        min_outer_loop = loop
            if min_outer_loop is None:
                return None
            for table_occurrence, column in join_condition.equivalence_class():
                if table_occurrence in inner_table_occurrences_for_loop:
                    existing = inner_column_loop_counts.get(column)
                    if existing is None or min_outer_loop < existing:
                        inner_column_loop_counts[column] = min_outer_loop
        if not inner_column_loop_counts:
            return None

        # FD pruning: if an inner table's PK columns are all present as
        # inner-side join key columns, non-PK columns from that table
        # are functionally determined and don't add independent key dimensions.
        pk_covered = self._pk_covered_tables(join_conditions, inner_table_occurrences_for_loop)

        active_loop_counts = {}
        for col, lc in inner_column_loop_counts.items():
            pruned = False
            for table in pk_covered:
                if col in table.columns() and self._is_fd_determined(table, col, pk_covered):
                    pruned = True
                    if self.debug_nested_loop:
                        print(f"      [NL FD prune] inner_col={col.name()} table={table.name()}")
                    break
            if not pruned:
                active_loop_counts[col] = lc

        if not active_loop_counts:
            active_loop_counts = dict(inner_column_loop_counts)

        if self.debug_nested_loop:
            for col, lc in active_loop_counts.items():
                print(f"      [NL join key] inner_col={col.name()} loop_count={lc:.0f}")

        # Interpolate between correlated (max) and independent (product) extremes,
        # bounding the product at outer_cardinality before interpolation.
        loop_values = list(active_loop_counts.values())
        loop_product = min(float(np.prod(loop_values)), outer_cardinality_estimate)
        loop_max = max(loop_values)
        loop_count = math.sqrt(loop_max * loop_product)

        if self.debug_nested_loop:
            print(f"      [NL loop_count] max={loop_max:.0f} product={loop_product:.0f} combined={loop_count:.0f} outer_card={outer_cardinality_estimate:.0f}")

        assert loop_count <= outer_cardinality_estimate and loop_count >= 0

        is_memoized = self._is_memoized_index_scan(nested_loop_join_expression.inner)
        if is_memoized:
            memoize_key_widths = [col.width() for col in active_loop_counts.keys()]
        else:
            memoize_key_widths = []

        non_memoized_ratio = (loop_count / outer_cardinality_estimate) if is_memoized else 1.0

        rescan_costs = self._cost_rescan(nested_loop_join_expression, nested_loop_join_expression.inner, loop_count, non_memoized_ratio, cardinality_mode=cardinality_mode)
        if rescan_costs is None:
            return None
        initial_inner_cost, inner_rescan_run_cost, inner_cardinality_estimate = rescan_costs

        # When memoization is active, compute the amortized per-rescan cost
        # using PG's cost_memoize_rescan model (memory-bounded cache capacity,
        # eviction costs, cache storage costs, and lookup overhead).
        if is_memoized and non_memoized_ratio < 1:
            if self._default_width is None:
                memo_width = nested_loop_join_expression.inner.query().width()
            else:
                memo_width = self._default_width
            rescan_startup, rescan_total = self._cost_memoize_rescan(
                initial_inner_cost, inner_rescan_run_cost,
                inner_cardinality_estimate, outer_cardinality_estimate,
                loop_count, memo_width, memoize_key_widths)
            memoized_rescan_run_cost = rescan_total - rescan_startup
            initial_inner_cost += rescan_startup
        else:
            memoized_rescan_run_cost = inner_rescan_run_cost

        inner_query = nested_loop_join_expression.inner.query()
        assert isinstance(inner_query, SPJQuery)

        inner_table_occurrences = inner_query.table_occurrences()
        inner_unique = self._is_inner_unique(inner_query, nested_loop_join_expression.join_condition, inner_table_occurrences)

        cost += initial_inner_cost
        self._last_has_indexed_join_quals = None
        if inner_unique:  # Also for semi and anti joins in PostgreSQL, but we don't have those yet
            cardinality_estimate = self.cardinality_estimator.estimate(nested_loop_join_expression.canonical_query(), cardinality_mode=cardinality_mode)
            if cardinality_estimate is None:
                return None
            if cardinality_estimate > outer_cardinality_estimate:
                outer_matched_rows = outer_cardinality_estimate
            else:
                outer_matched_rows = cardinality_estimate
            outer_unmatched_rows = outer_cardinality_estimate - outer_matched_rows
            match_count = 1  # because we only cover the inner unique case here
            # TODO: this is what PostgreSQL calculates, but with the "fuzz factor" 2 it is quite silly here for only the inner unique case, since it completely cancels the discount because of early stopping
            if self._improved and match_count == 1:
                inner_scan_fraction = 0.5
            else:
                inner_scan_fraction = 2 / (match_count + 1)
            ntuples = outer_matched_rows * inner_cardinality_estimate * inner_scan_fraction
            has_indexed_join_quals = self._has_indexed_join_quals(nested_loop_join_expression)
            self._last_has_indexed_join_quals = has_indexed_join_quals
            if has_indexed_join_quals:
                if outer_matched_rows > 1:
                    cost += (outer_matched_rows - 1) * memoized_rescan_run_cost * inner_scan_fraction
                cost += outer_unmatched_rows * memoized_rescan_run_cost / inner_cardinality_estimate
            else:
                ntuples += outer_unmatched_rows * inner_cardinality_estimate

                if outer_unmatched_rows >= 1:
                    outer_unmatched_rows -= 1
                else:
                    outer_matched_rows -= 1

                if outer_matched_rows > 0:
                    cost += outer_matched_rows * memoized_rescan_run_cost * inner_scan_fraction

                if outer_unmatched_rows > 0:
                    cost += outer_unmatched_rows * memoized_rescan_run_cost

        else:
            # First scan at full cost, subsequent scans at the amortized rate
            # (which folds in hit/miss ratio, eviction, and cache storage costs
            # when memoization is active).  When non_memoized_ratio == 1 this
            # collapses to outer_cardinality_estimate * inner_rescan_run_cost.
            cost += inner_rescan_run_cost + (outer_cardinality_estimate - 1) * memoized_rescan_run_cost
            ntuples = outer_cardinality_estimate * inner_cardinality_estimate

        qual_cost_per_tuple = self._predicate_cost_per_tuple(nested_loop_join_expression.join_condition)
        cpu_per_tuple = self._postgresql_configuration.cpu_tuple_cost + self._probe_cache_penalty * qual_cost_per_tuple
        cost += ntuples * cpu_per_tuple

        if self.debug_nested_loop:
            inner_label = type(nested_loop_join_expression.inner).__name__.replace("Expression", "")
            indexed_jq = getattr(self, "_last_has_indexed_join_quals", None)
            print(
                f"    [NL debug] inner={inner_label} outer_card={outer_cardinality_estimate:.0f} inner_per_scan={inner_cardinality_estimate:.0f} "
                f"loop_count={loop_count:.2f} non_memoized_ratio={non_memoized_ratio:.3f} initial_inner={initial_inner_cost:.2f} "
                f"inner_rescan_run={inner_rescan_run_cost:.2f} inner_unique={inner_unique} indexed_join_quals={indexed_jq} ntuples={ntuples:.0f} cpu_per_tuple={cpu_per_tuple:.5f} total={cost:.2f}"
            )

        return cost

    @staticmethod
    def _is_inner_unique(inner_query, join_condition: Predicate, inner_table_occurrences: Set[TableOccurrence]) -> bool:
        # True iff some inner table_occurrence's own outer-bound columns cover a
        # UK of its table; columns_are_unique then verifies that every other
        # inner table is downstream-unique from that anchor via the inner join
        # graph. Trying every candidate anchor (instead of first-wins) is what
        # makes multi-table inners detectable.
        if not isinstance(inner_query, SPJQuery):
            return False
        inner_join_columns = PostgreSQLCostModel._join_columns_for_table_occurrences(join_condition, inner_table_occurrences)
        columns_by_table_occurrence: dict = {}
        for table_occurrence, column in inner_join_columns:
            columns_by_table_occurrence.setdefault(table_occurrence, []).append(column)
        return any(
            inner_query.columns_are_unique(anchor, cols)
            for anchor, cols in columns_by_table_occurrence.items()
        )

    @staticmethod
    def _join_columns_for_table_occurrences(join_condition: Predicate, table_occurrences: Set[TableOccurrence]) -> List[Tuple[TableOccurrence, Column]]:
        """Return (table_occurrence, column) pairs from join_condition whose table_occurrence is in table_occurrences."""
        if isinstance(join_condition, Join):
            join_conditions = [join_condition]
        elif isinstance(join_condition, Conjunction):
            join_conditions = join_condition.predicates()
        else:
            raise ValueError("Unsupported join condition type: %s" % type(join_condition))
        result = []
        for jc in join_conditions:
            for table_occurrence, column in jc.equivalence_class():
                if table_occurrence in table_occurrences:
                    result.append((table_occurrence, column))
        return result

    @staticmethod
    def _is_index_scan(relational_algebra_expression: RelationalAlgebraExpression) -> bool:
        if isinstance(relational_algebra_expression, IndexScanExpression):
            return True
        elif isinstance(relational_algebra_expression, GroupRelationalAlgebraExpression) and relational_algebra_expression.requirements.force_index_scan is not None:
            return True
        else:
            return False

    def _has_indexed_join_quals(self, nested_loop_join_expression: NestedLoopJoinExpression) -> bool:
        # Mirrors PostgreSQL's `has_indexed_join_quals`: true when the inner path is an index scan whose index
        # condition encompasses every join clause of the surrounding nested loop. In that case an outer row that
        # has no matching inner tuple terminates the inner scan almost immediately (cost ~ 1/inner_cardinality of
        # a full rescan) instead of paying the full inner scan cost — a very large effect on unique PK joins.
        inner = nested_loop_join_expression.inner
        if isinstance(inner, IndexScanExpression):
            if not inner.join_clause:
                return False
            index = inner.index
            inner_table_occurrences = inner.query().table_occurrences()
        elif isinstance(inner, GroupRelationalAlgebraExpression) and inner.requirements.force_index_scan is not None:
            index = inner.requirements.force_index_scan[0]
            inner_table_occurrences = inner.query().table_occurrences()
        else:
            return False
        if index is None:
            return False
        index_columns = set(index.columns())
        join_condition = nested_loop_join_expression.join_condition
        if isinstance(join_condition, Join):
            join_clauses = [join_condition]
        elif isinstance(join_condition, Conjunction):
            join_clauses = join_condition.predicates()
        else:
            return False
        for clause in join_clauses:
            if not isinstance(clause, Join):
                return False
            covered = False
            for table_occurrence, column in clause.equivalence_class():
                if table_occurrence in inner_table_occurrences and column in index_columns:
                    covered = True
                    break
            if not covered:
                return False
        return True

    @staticmethod
    def _is_memoized_index_scan(relational_algebra_expression: RelationalAlgebraExpression) -> bool:
        if isinstance(relational_algebra_expression, IndexScanExpression) and relational_algebra_expression.memoized:
            return True
        elif isinstance(relational_algebra_expression, GroupRelationalAlgebraExpression) and relational_algebra_expression.requirements.force_index_scan is not None and relational_algebra_expression.requirements.force_index_scan[1]:
            return True
        else:
            return False

    @staticmethod
    def _prefix_bound_query(parent_query: SPJQuery, inner_table_occurrence: TableOccurrence, prefix_columns: Set[Column]) -> Optional[SPJQuery]:
        """Query whose cardinality equals the index entries walked across all probes of
        an index nested loop: the parent query constrained only by what the index prefix
        enforces, i.e. join conjuncts and equality predicates on prefix columns of the
        inner table. Join conjuncts and predicates outside the prefix are applied as
        filters after the index walk and must not constrain the bound. Returns None if
        no join conjunct is covered by the prefix, since the reduced query would not
        connect the inner table."""
        if len(parent_query.non_equi_join_predicates()) > 0:
            return None
        covered = any(table_occurrence == inner_table_occurrence and column in prefix_columns
                      for join in parent_query.joins()
                      for table_occurrence, column in join.equivalence_class())
        if not covered:
            return None
        predicate = inner_table_occurrence.predicate()
        if isinstance(predicate, Conjunction):
            predicates = predicate.predicates()
        else:
            predicates = [predicate]
        kept_predicates = [p for p in predicates
                           if isinstance(p, SimplePredicate) and p.operator() == COMPARISON_OPERATOR_EQ and p.column() in prefix_columns]
        new_inner_table_occurrence = TableOccurrence(inner_table_occurrence.table(), inner_table_occurrence.alias())
        if len(kept_predicates) == 0:
            new_predicate = TruePredicate()
        elif len(kept_predicates) == 1:
            new_predicate = kept_predicates[0]
        else:
            new_predicate = Conjunction(kept_predicates)
        new_predicate = new_predicate.replace({inner_table_occurrence: new_inner_table_occurrence})
        new_inner_table_occurrence.set_predicate(new_predicate)
        table_occurrences = [new_inner_table_occurrence if table_occurrence == inner_table_occurrence else table_occurrence
                             for table_occurrence in parent_query.table_occurrences()]
        joins = []
        for join in parent_query.joins():
            equivalence_class = []
            for table_occurrence, column in join.equivalence_class():
                if table_occurrence == inner_table_occurrence:
                    if column in prefix_columns:
                        equivalence_class.append((new_inner_table_occurrence, column))
                else:
                    equivalence_class.append((table_occurrence, column))
            if len(equivalence_class) > 1:
                joins.append(Join(equivalence_class))
        return SPJQuery(table_occurrences, joins, [])

    def _cost_rescan(self, parent_relational_algebra_expression: RelationalAlgebraExpression, relational_algebra_expression: RelationalAlgebraExpression, loop_count: int, non_memoized_ratio: float, cardinality_mode: CardinalityMode) -> Optional[Tuple[float, float]]:
        # Returning upfront costs and per rescan costs
        if self._is_index_scan(relational_algebra_expression):
            if isinstance(relational_algebra_expression, IndexScanExpression):
                index_scan_expression = relational_algebra_expression
            elif isinstance(relational_algebra_expression, GroupRelationalAlgebraExpression):
                table_occurrences = relational_algebra_expression.query().table_occurrences()
                assert len(table_occurrences) == 1
                table_occurrence = list(table_occurrences)[0]
                index, is_memoized = relational_algebra_expression.requirements.force_index_scan
                index_scan_expression = IndexScanExpression(table_occurrence, index, is_memoized, True, query=relational_algebra_expression.query())
            parent_query = parent_relational_algebra_expression.query()
            assert isinstance(parent_query, SPJQuery)
            min_cardinality = self.cardinality_estimator.estimate(parent_query, cardinality_mode=CardinalityMode.MIN)
            if min_cardinality is None:
                min_cardinality = 0.0
            # Determine which inner columns are covered by join conditions or equality
            # predicates so that _bt_cost_estimate uses the full useful index prefix.
            matched_columns = None
            if isinstance(parent_relational_algebra_expression, NestedLoopJoinExpression):
                inner_table_occurrences = relational_algebra_expression.query().table_occurrences()
                matched_columns = {col for _, col in self._join_columns_for_table_occurrences(parent_relational_algebra_expression.join_condition, inner_table_occurrences)}
                # Also include equality predicate columns from the inner table so that
                # composite indexes like (join_col, predicate_col) get full credit for
                # their selectivity when the predicate is a constant equality.
                inner_predicate = index_scan_expression.table_occurrence.predicate()
                if isinstance(inner_predicate, Conjunction):
                    for p in inner_predicate.predicates():
                        if isinstance(p, SimplePredicate) and p.operator() == COMPARISON_OPERATOR_EQ:
                            matched_columns.add(p.column())
                elif isinstance(inner_predicate, SimplePredicate) and inner_predicate.operator() == COMPARISON_OPERATOR_EQ:
                    matched_columns.add(inner_predicate.column())
            # Bounds for num_index_tuples in _bt_cost_estimate: the parent query constrained only by
            # the index prefix, see _prefix_bound_query.
            unfiltered_parent_query = None
            if matched_columns is not None and index_scan_expression.index is not None:
                prefix_columns = set(self._matched_prefix_columns_for_index(index_scan_expression.index, matched_columns))
                unfiltered_parent_query = self._prefix_bound_query(parent_query, index_scan_expression.table_occurrence, prefix_columns)
            if unfiltered_parent_query is None:
                unfiltered_parent_query = parent_query.remove_predicate(index_scan_expression.table_occurrence)
            min_unfiltered_cardinality = self.cardinality_estimator.estimate(unfiltered_parent_query, cardinality_mode=CardinalityMode.MIN)
            if min_unfiltered_cardinality is None:
                min_unfiltered_cardinality = 0.0
            min_unfiltered_cardinality = max(min_cardinality, min_unfiltered_cardinality)
            min_unfiltered_cardinality = min_unfiltered_cardinality * non_memoized_ratio
            max_unfiltered_cardinality = self.cardinality_estimator.estimate(unfiltered_parent_query, cardinality_mode=CardinalityMode.MAX)
            if max_unfiltered_cardinality is not None:
                max_unfiltered_cardinality = max_unfiltered_cardinality * non_memoized_ratio
            average_result = self._index_scan_cost(index_scan_expression,
                                                   loop_count=loop_count,
                                                   include_startup_cost=True,
                                                   min_unfiltered_cardinality=min_unfiltered_cardinality,
                                                   max_unfiltered_cardinality=max_unfiltered_cardinality,
                                                   matched_columns=matched_columns,
                                                   cardinality_mode=cardinality_mode)
            if average_result is None:
                return None
            average_cost, cardinality = average_result
            average_cost = max(0, average_cost - self._index_scan_base_cost / max(loop_count, 1))
            return 0, average_cost, cardinality
        else:
            # If it is not an index scan we assume that PostgreSQL will materialize the result
            cardinality_estimate = self.cardinality_estimator.estimate(relational_algebra_expression.canonical_query(), cardinality_mode=cardinality_mode)
            if cardinality_estimate is None:
                return None
            if self._default_width is None:
                tuple_width = relational_algebra_expression.query().width()
            else:
                tuple_width = self._default_width
            # Material's incremental one-time store overhead; per-scan emit is paid via cpu_cost on each of the N scans counted in _nested_loop_join_cost
            initial_cpu_cost = self._probe_cache_penalty * cardinality_estimate * self._postgresql_configuration.cpu_operator_cost

            # Reimplementation of cost_rescan
            cpu_cost = self._probe_cache_penalty * cardinality_estimate * self._postgresql_configuration.cpu_operator_cost
            nbytes = self._relation_byte_size(cardinality_estimate, tuple_width)
            work_mem_bytes = int(self._postgresql_configuration.work_mem * 1024)
            if nbytes > work_mem_bytes:
                npages = math.ceil(nbytes / self._postgresql_configuration.block_size)
                io_cost = npages * self._postgresql_configuration.seq_page_cost
                # Again from cost_material
                initial_io_cost = npages * self._postgresql_configuration.seq_page_cost
            else:
                io_cost = 0.0
                initial_io_cost = 0.0
            return initial_cpu_cost + initial_io_cost, cpu_cost + io_cost, cardinality_estimate

    def _cost_memoize_rescan(self, input_startup_cost: float, input_total_cost: float,
                             tuples: float, calls: float, ndistinct: float,
                             width: int, key_widths: list) -> Tuple[float, float]:
        """Reimplementation of PostgreSQL's cost_memoize_rescan (costsize.c).

        Returns (rescan_startup_cost, rescan_total_cost) representing the
        amortized cost of a single rescan through a Memoize node, accounting
        for cache capacity, hit ratio, eviction overhead, and storage costs.
        """
        if ndistinct <= 0 or calls <= 0:
            return input_startup_cost, input_total_cost

        # PG 18 struct sizes on 64-bit: MemoizeEntry=24, MemoizeKey=24, MemoizeTuple=16
        sizeof_memoize_entry = 24
        sizeof_memoize_key = 24
        sizeof_memoize_tuple = 16

        hash_mem_bytes = self._get_hash_memory_limit()

        size_of_heap_tuple_header = self._size_of_heap_tuple_header()
        est_entry_bytes = tuples * (self._max_align(width) + self._max_align(size_of_heap_tuple_header))
        est_entry_bytes += sizeof_memoize_entry + sizeof_memoize_key + sizeof_memoize_tuple * tuples
        for kw in key_widths:
            est_entry_bytes += kw

        est_cache_entries = max(math.floor(hash_mem_bytes / max(est_entry_bytes, 1)), 1)

        evict_ratio = 1.0 - min(est_cache_entries, ndistinct) / ndistinct

        hit_ratio = ((calls - ndistinct) / calls) * (est_cache_entries / max(ndistinct, est_cache_entries))
        hit_ratio = max(0.0, min(hit_ratio, 1.0))

        cpu_operator_cost = self._postgresql_configuration.cpu_operator_cost
        cpu_tuple_cost = self._postgresql_configuration.cpu_tuple_cost

        total_cost = input_total_cost * (1.0 - hit_ratio) + cpu_operator_cost
        total_cost += cpu_tuple_cost * evict_ratio
        total_cost += (cpu_operator_cost / 10.0) * evict_ratio * tuples
        total_cost += cpu_tuple_cost + cpu_operator_cost * tuples

        startup_cost = input_startup_cost * (1.0 - hit_ratio) + cpu_tuple_cost

        return startup_cost, total_cost

    def _sequential_scan_cost(self, sequential_scan_expression: SequentialScanExpression) -> float:
        table = sequential_scan_expression.table_occurrence.table()
        baserel_tuples = table.cardinality()
        spc_seq_page_cost = self._postgresql_configuration.seq_page_cost  # get_tablespace_page_costs in PostgreSQL, TODO: handle multiple tablespaces
        baserel_pages = self._baserel_pages(table)
        disk_run_cost = spc_seq_page_cost * baserel_pages
        qp_qual_cost_per_tuple = self._predicate_cost_per_tuple(sequential_scan_expression.table_occurrence.predicate())  # get_restriction_qual_cost -> cost_qual_eval -> add_function_cost in PostgreSQL
        cpu_per_tuple = self._postgresql_configuration.cpu_tuple_cost + qp_qual_cost_per_tuple
        cpu_run_cost = cpu_per_tuple * baserel_tuples
        return disk_run_cost + cpu_run_cost

    def _baserel_pages(self, table: Table) -> int:
        return math.ceil(table.table_size() / self._postgresql_configuration.block_size)

    def _index_scan_cost(self,
                         index_scan_expression: IndexScanExpression,
                         loop_count: int = 1,
                         include_startup_cost: bool = True,
                         min_unfiltered_cardinality: Optional[float] = None,
                         max_unfiltered_cardinality: Optional[float] = None,
                         predicate: Optional[Predicate] = None,
                         matched_columns: Optional[Set[Column]] = None,
                         cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[Tuple[float, int]]:
        table = index_scan_expression.table_occurrence.table()

        run_cost = 0

        index_total_cost, index_selectivity, index_correlation = self._bt_cost_estimate(index_scan_expression.index, loop_count, include_startup_cost, min_unfiltered_cardinality, max_unfiltered_cardinality, cardinality_mode, table, matched_columns)

        run_cost += index_total_cost

        tuples_fetched = self._clamp_row_est(index_selectivity * table.cardinality())
        base_rel_pages = self._baserel_pages(table)
        if loop_count > 1:
            max_pages_fetched = self._index_pages_fetched(tuples_fetched * loop_count, base_rel_pages)
            max_io_cost = max_pages_fetched * self._postgresql_configuration.random_page_cost / loop_count
            min_pages_fetched = self._index_pages_fetched(math.ceil(index_selectivity * base_rel_pages) * loop_count, base_rel_pages)
            min_io_cost = min_pages_fetched * self._postgresql_configuration.random_page_cost / loop_count
        else:
            max_pages_fetched = self._index_pages_fetched(tuples_fetched, base_rel_pages)
            max_io_cost = max_pages_fetched * self._postgresql_configuration.random_page_cost

            min_pages_fetched = math.ceil(index_selectivity * base_rel_pages)
            if min_pages_fetched > 0:
                min_io_cost = self._postgresql_configuration.random_page_cost
                if min_pages_fetched > 1:
                    min_io_cost += (min_pages_fetched - 1) * self._postgresql_configuration.seq_page_cost
            else:
                min_io_cost = 0.0

        if index_scan_expression.index is not None and index_scan_expression.index.covers_columns(table.columns()):
            # Static, post-vacuum DB: all heap pages are all-visible, so IOS skips every heap fetch.
            max_io_cost = 0.0
            min_io_cost = 0.0

        c_squared = index_correlation ** 2
        if min_io_cost > max_io_cost + 1e-6:
            raise ValueError("Inconsistent I/O cost estimates: min_io_cost=%f, max_io_cost=%f" % (min_io_cost, max_io_cost))
        run_cost += max_io_cost + c_squared * (min_io_cost - max_io_cost)

        if predicate is None:
            predicate = index_scan_expression.table_occurrence.predicate()
        # Equality predicates whose columns are covered by the index prefix are
        # satisfied during the index seek and should not be charged again here.
        if matched_columns is not None and len(matched_columns) > 0:
            predicate = self._strip_index_covered_predicates(predicate, matched_columns)
        cpu_per_tuple = self._postgresql_configuration.cpu_tuple_cost + self._predicate_cost_per_tuple(predicate)
        run_cost += cpu_per_tuple * tuples_fetched
        return run_cost, tuples_fetched

    def _index_pages_fetched(self, tuples_fetched: float, pages: float) -> float:
        # Assuming b >= T
        T = max(pages, 1.0)
        pages_fetched = 2 * T * tuples_fetched / (2 * T + tuples_fetched)
        if pages_fetched >= T:
            pages_fetched = T
        else:
            pages_fetched = math.ceil(pages_fetched)
        return pages_fetched

    @staticmethod
    def _matched_prefix_columns_for_index(index: Index, matched_columns: Optional[Set[Column]]) -> List[Column]:
        """The leading index columns covered by matched_columns (all leading columns if
        matched_columns is None) that carry statistics — the boundary prefix that
        determines num_index_tuples in _bt_cost_estimate."""
        matched_prefix_columns: List[Column] = []
        for index_column in index.columns():
            if matched_columns is not None and index_column not in matched_columns:
                break
            if index_column.distinct_count() is not None:
                matched_prefix_columns.append(index_column)
        return matched_prefix_columns

    def _bt_cost_estimate(self,
                          index: Index,
                          loop_count: int,
                          include_startup_cost: bool,
                          min_unfiltered_cardinality: Optional[float],
                          max_unfiltered_cardinality: Optional[float],
                          cardinality_mode: CardinalityMode,
                          table: Table,
                          matched_columns: Optional[Set[Column]] = None) -> Tuple[float, float, float]:
        index_columns = index.columns()
        matched_prefix_columns = self._matched_prefix_columns_for_index(index, matched_columns)
        matched_prefix_length = len(matched_prefix_columns)
        index_tuples = index.tuples()
        index_pages = index.pages()

        if matched_prefix_length == 0:
            # No prefix match: each probe deterministically scans the whole
            # table. The min/max_unfiltered bounds cap per-rescan join output,
            # not per-rescan index entries read, so they don't apply here.
            num_index_tuples = float(table.cardinality())
        else:
            # Index entries walked per probe: the prefix-bound cardinality of the parent query per
            # probe for MIN and MAX, the statistics-based estimate clamped to that range for MEAN.
            scans = max(loop_count, 1)
            min_num_index_tuples = 0.0 if min_unfiltered_cardinality is None else min_unfiltered_cardinality / scans
            max_num_index_tuples = float(index_tuples) if max_unfiltered_cardinality is None else max_unfiltered_cardinality / scans
            if cardinality_mode == CardinalityMode.MIN:
                num_index_tuples = min_num_index_tuples
            elif cardinality_mode == CardinalityMode.MAX:
                num_index_tuples = max_num_index_tuples
            else:
                distinct_combinations = index.prefix_distinct_count(matched_prefix_length)
                if distinct_combinations < 1:
                    distinct_combinations = 1
                num_index_tuples = index_tuples / distinct_combinations
                null_fractions = []
                for index_column in index_columns[:matched_prefix_length]:
                    null_fraction = index_column.null_fraction()
                    if null_fraction is not None:
                        null_fractions.append(null_fraction)
                if len(null_fractions) > 0:
                    null_fraction = 1 - np.prod([1 - nf for nf in null_fractions])
                    num_index_tuples = num_index_tuples * (1 - null_fraction)
                num_index_tuples = min(max(num_index_tuples, min_num_index_tuples), max_num_index_tuples)

            # Per-key degree bounds of the index prefix. They only apply when the prefix columns are
            # bound by an outer join condition or equality predicate; for standalone IndexScans
            # (matched_columns=None) selectivity comes from a WHERE predicate, not a per-key lookup.
            if matched_columns is not None:
                max_degree = self.composite_max_degree(table, matched_prefix_columns)
                if max_degree is not None and num_index_tuples > max_degree:
                    num_index_tuples = float(max_degree)
                min_degree = self.composite_min_degree_lower_bound(table, matched_prefix_columns)
                if min_degree is not None and num_index_tuples < min_degree:
                    num_index_tuples = float(min_degree)

        if num_index_tuples > index_tuples:
            num_index_tuples = index_tuples

        index_selectivity = num_index_tuples / index_tuples

        total_cost = 0

        # Not necessary to use these variables as is, but we keep them to clarify the mapping to PostgreSQL code
        num_sa_scans = 1

        # Reimplemented from genericcostestimate
        if index_pages > 1 and index_tuples > 1:
            num_index_pages = math.ceil(num_index_tuples * index_pages / index_tuples)
        else:
            num_index_pages = 1

        num_scans = num_sa_scans * loop_count
        if num_scans > 1:
            pages_fetched = self._index_pages_fetched(num_index_pages * num_scans, index_pages)
            total_cost += pages_fetched * self._postgresql_configuration.random_page_cost / loop_count
        else:
            total_cost += num_index_pages * self._postgresql_configuration.random_page_cost
        total_cost += num_sa_scans * num_index_tuples * self._postgresql_configuration.cpu_index_tuple_cost

        # Back to btcostestimate
        if index_tuples > 1:
            descent_cost = math.ceil(math.log(index_tuples) / math.log(2)) * self._postgresql_configuration.cpu_operator_cost
            if include_startup_cost:
                total_cost += descent_cost
            if num_sa_scans > 1:
                total_cost += (num_sa_scans - 1) * descent_cost

        default_page_cpu_multiplier = 50
        descent_cost = (index.level() + 2) * default_page_cpu_multiplier * self._postgresql_configuration.cpu_operator_cost  # index->tree_height = index.level() + 1
        if include_startup_cost:
            total_cost += descent_cost
        if num_sa_scans > 1:
            total_cost += (num_sa_scans - 1) * descent_cost

        correlation = index.columns()[0].correlation()
        if correlation is None:
            correlation = 0.0
        # btcostestimate discounts the first column's correlation for multi-column indexes.
        if len(index_columns) > 1:
            correlation *= 0.75

        return total_cost, index_selectivity, correlation

    def _estimate_num_groups(self, table_occurrence: TableOccurrence, column: Column, max_table_occurrence_cardinality: Optional[float], cardinality_mode: CardinalityMode) -> Optional[int]:
        column_distinct_count = column.distinct_count()
        if column_distinct_count is None:
            return None
        column_distinct_count = max(1, column_distinct_count)
        table_cardinality = table_occurrence.table().cardinality()
        table_occurrence_query = SPJQuery([table_occurrence], [], [])
        table_occurrence_cardinality = self.cardinality_estimator.estimate(table_occurrence_query, cardinality_mode=cardinality_mode)
        if table_occurrence_cardinality is None or table_occurrence_cardinality > table_cardinality:
            table_occurrence_cardinality = table_cardinality
        if max_table_occurrence_cardinality is not None and table_occurrence_cardinality > max_table_occurrence_cardinality:
            table_occurrence_cardinality = max_table_occurrence_cardinality
        num_distinct = column_distinct_count * (1 - ((table_cardinality - table_occurrence_cardinality) / table_cardinality) ** (table_cardinality / column_distinct_count))
        if num_distinct > column_distinct_count:
            num_distinct = column_distinct_count
        if num_distinct > table_occurrence_cardinality:
            num_distinct = table_occurrence_cardinality
        return num_distinct

    def _where_clause_index_scan_cost(self, index_scan_expression: IndexScanExpression, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        table_occurrence = index_scan_expression.table_occurrence
        table = table_occurrence.table()
        index = index_scan_expression.index
        index_columns = index.columns()
        min_cardinality = self.cardinality_estimator.estimate(index_scan_expression.query(), cardinality_mode=CardinalityMode.MIN)
        if min_cardinality is None:
            min_cardinality = 0.0
        predicate = table_occurrence.predicate()
        if isinstance(predicate, Conjunction):
            predicates = predicate.predicates()
        else:
            predicates = [predicate]
        # Predicates not on index columns are evaluated as the heap filter and stay
        # charged per fetched tuple in _index_scan_cost.
        filter_predicates = []
        for predicate in predicates:
            if not isinstance(predicate, SimplePredicate) or predicate.column() not in index_columns:
                filter_predicates.append(predicate)
        if len(filter_predicates) == 0:
            filter_predicate = TruePredicate()
        elif len(filter_predicates) == 1:
            filter_predicate = filter_predicates[0]
        else:
            filter_predicate = Conjunction(filter_predicates)
        # num_index_tuples counts the index entries matching the boundary quals: the
        # predicates on a leading consecutive prefix of the index columns (equalities,
        # plus range predicates on the column that ends the prefix). The bound query
        # keeps exactly those predicates, so its cardinality equals that entry count.
        boundary_operators = (COMPARISON_OPERATOR_EQ, COMPARISON_OPERATOR_LT, COMPARISON_OPERATOR_LTE, COMPARISON_OPERATOR_GT, COMPARISON_OPERATOR_GTE, COMPARISON_OPERATOR_IN)
        boundary_predicates = []
        for index_column in index_columns:
            column_predicates = [p for p in predicates
                                 if isinstance(p, SimplePredicate) and p.column() == index_column and p.operator() in boundary_operators]
            if len(column_predicates) == 0:
                break
            boundary_predicates.extend(column_predicates)
            if any(p.operator() != COMPARISON_OPERATOR_EQ for p in column_predicates):
                break
        bound_table_occurrence = TableOccurrence(table, table_occurrence.alias())
        if len(boundary_predicates) == 0:
            bound_predicate = TruePredicate()
        elif len(boundary_predicates) == 1:
            bound_predicate = boundary_predicates[0]
        else:
            bound_predicate = Conjunction(boundary_predicates)
        bound_predicate = bound_predicate.replace({table_occurrence: bound_table_occurrence})
        bound_table_occurrence.set_predicate(bound_predicate)
        bound_query = SPJQuery([bound_table_occurrence], [], [])
        min_unfiltered_cardinality = self.cardinality_estimator.estimate(bound_query, cardinality_mode=CardinalityMode.MIN)
        if min_unfiltered_cardinality is None:
            min_unfiltered_cardinality = 0.0
        min_unfiltered_cardinality = max(min_cardinality, min_unfiltered_cardinality)
        max_unfiltered_cardinality = self.cardinality_estimator.estimate(bound_query, cardinality_mode=CardinalityMode.MAX)
        index_scan_result = self._index_scan_cost(index_scan_expression, min_unfiltered_cardinality=min_unfiltered_cardinality, max_unfiltered_cardinality=max_unfiltered_cardinality, predicate=filter_predicate, cardinality_mode=cardinality_mode)
        if index_scan_result is None:
            return None
        index_scan_cost, _ = index_scan_result
        return index_scan_cost











