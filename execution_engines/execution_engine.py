import logging
import math
import re
import threading
from abc import abstractmethod
from typing import Dict, Tuple, List, Optional, Set
import datetime

from cardinality_estimators.cardinality_range import CardinalityRange
from execution_engines.execution_data import ExecutionData
from queries.benchmark_query import BenchmarkQuery
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.query_data.parse import parse_table_predicate
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.bitmap_index_scan_expression import BitmapIndexScanExpression
from relational_algebra_expressions.scan_expressions.index_only_scan_expression import IndexOnlyScanExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.index import Index
from schemas.schema import Schema
from schemas.table import Table


class ExecutionEngine:
    PASSTHROUGH_NODE_TYPES = {"Aggregate", "Gather", "Gather Merge", "Hash", "Incremental Sort", "Limit", "Materialize", "Memoize", "Sort", "Unique"}

    def __init__(self, name: str, schema: Schema, set_commands: List[str], max_parallel_workers_per_gather: int, timeout: Optional[float] = None, verify_plan: bool = True, verify_cardinalities: bool = False):
        self.name: str = name
        self.schema: Schema = schema
        self.set_commands: List[str] = set_commands
        self.max_parallel_workers_per_gather: int = max_parallel_workers_per_gather
        self.set_commands.append("SET max_parallel_workers_per_gather = %d;" % self.max_parallel_workers_per_gather)
        self.timeout: Optional[float] = timeout  # in milliseconds
        self.verify_plan: bool = verify_plan
        self.verify_cardinalities: bool = verify_cardinalities
        self._parallelism_uncertainty = 1

    @staticmethod
    def _collect_plan_index_counts(plan: RelationalAlgebraExpression) -> Dict[Tuple[Table, Index], int]:
        """Count how many times each (Table, Index) appears in the plan."""
        counts: Dict[Tuple[Table, Index], int] = {}
        if (isinstance(plan, IndexScanExpression) or isinstance(plan, IndexOnlyScanExpression)) and plan.index is not None:
            key = (plan.table_occurrence.table(), plan.index)
            counts[key] = counts.get(key, 0) + 1
        for child in plan.children:
            for key, count in ExecutionEngine._collect_plan_index_counts(child).items():
                counts[key] = counts.get(key, 0) + count
        return counts

    @staticmethod
    def _map_indexes_to_outer_queries(plan: RelationalAlgebraExpression, unique_indexes: Set[Tuple[Table, Index]]) -> Dict[Tuple[Table, Index], Query]:
        """For each NestedLoopJoin whose inner is an IndexScan with a unique index, map (Table, Index) -> outer query."""
        mapping: Dict[Tuple[Table, Index], Query] = {}
        if isinstance(plan, NestedLoopJoinExpression):
            inner = plan.inner
            # Walk through Materialize-like wrappers (single-child non-join, non-scan nodes)
            while not isinstance(inner, (ScanExpression, JoinExpression)) and len(inner.children) == 1:
                inner = inner.children[0]
            if (isinstance(inner, IndexScanExpression) or isinstance(inner, IndexOnlyScanExpression)) and inner.index is not None:
                key = (inner.table_occurrence.table(), inner.index)
                if key in unique_indexes:
                    mapping[key] = plan.outer.query()
        for child in plan.children:
            mapping.update(ExecutionEngine._map_indexes_to_outer_queries(child, unique_indexes))
        return mapping

    def _snapshot_index_stats(self, connection, index_keys: Set[Tuple[Table, Index]]) -> Dict[Tuple[Table, Index], int]:
        """Query pg_stat_all_indexes for idx_scan counts."""
        if not index_keys:
            return {}
        cursor = connection.cursor()
        conditions = " OR ".join(
            "(relname = '%s' AND indexrelname = '%s')" % (table.name(), index.name())
            for table, index in index_keys
        )
        cursor.execute("SELECT relname, indexrelname, idx_scan FROM pg_stat_all_indexes WHERE schemaname = '%s' AND (%s)" % (self.schema.schema_name(), conditions))
        key_lookup = {(table.name(), index.name()): (table, index) for table, index in index_keys}
        result = {key_lookup[(row[0], row[1])]: row[2] for row in cursor.fetchall()}
        cursor.close()
        return result

    def execute(self, query: Query, plan: Optional[RelationalAlgebraExpression], analyze: bool = True, collect_cardinality_estimates: bool = False, verbose: bool = False, benchmark_query: Optional[BenchmarkQuery] = None, set_commands: Optional[List[str]] = None) -> Optional[ExecutionData]:
        connection = self.schema.connection()

        # Snapshot index stats before execution
        if plan is not None and analyze:
            index_counts = self._collect_plan_index_counts(plan)
            unique_indexes = {key for key, count in index_counts.items() if count == 1}
            index_to_outer_query = self._map_indexes_to_outer_queries(plan, unique_indexes)
            index_keys = set(index_to_outer_query.keys())
            clear_cursor = connection.cursor()
            clear_cursor.execute("SELECT pg_stat_clear_snapshot();")
            clear_cursor.close()
            stats_before = self._snapshot_index_stats(connection, index_keys)
        else:
            index_to_outer_query = {}
            index_keys = set()
            stats_before = {}

        # Start a poll thread to capture index stats shortly before timeout
        poll_stats = [None]
        poll_stop = threading.Event()
        if stats_before and self.timeout is not None and self.timeout > 1000:
            poll_connection = self.schema.connection()
            poll_delay = self.timeout / 1000.0 - 1.0  # 1 second before timeout

            def poll_index_stats():
                if poll_stop.wait(poll_delay):
                    return  # Query finished before poll time
                cursor = poll_connection.cursor()
                cursor.execute("SELECT pg_stat_clear_snapshot();")
                cursor.close()
                poll_stats[0] = self._snapshot_index_stats(poll_connection, index_keys)

            poll_thread = threading.Thread(target=poll_index_stats, daemon=True)
            poll_thread.start()
        else:
            poll_thread = None

        cursor = connection.cursor()
        self._set(cursor)
        if set_commands is not None:
            # Per-call settings, applied after the engine's own.
            for set_command in set_commands:
                cursor.execute(set_command)
        explain_string = "EXPLAIN (ANALYZE %s, VERBOSE TRUE, FORMAT JSON)" % ("TRUE" if analyze else "FALSE")
        if benchmark_query is None:
            query_text = query.query_text()
            query_name = None
        else:
            query_text = benchmark_query.query_text
            query_name = benchmark_query.query_name()
        execution_query = self.execution_query(query_text, plan, explain_string)
        if verbose:
            print(execution_query)
        result = self._execute_query(cursor, execution_query)

        # Stop poll thread
        if poll_thread is not None:
            poll_stop.set()
            poll_thread.join()

        # Snapshot index stats after execution
        if stats_before:
            if result is not None:
                # Query succeeded: flush on same backend for exact stats
                try:
                    flush_cursor = connection.cursor()
                    flush_cursor.execute("SET statement_timeout = 0;")
                    flush_cursor.execute("SELECT pg_stat_force_next_flush();")
                    flush_cursor.execute("SELECT pg_stat_clear_snapshot();")
                    flush_cursor.close()
                    stats_after = self._snapshot_index_stats(connection, index_keys)
                except Exception:
                    connection.rollback()
                    stats_after = {}
                nested_loop_executions = self._executed_nested_loop_executions(result[0]["Plan"], index_keys)
            elif poll_stats[0] is not None:
                # Query timed out: use the pre-timeout poll snapshot
                stats_after = poll_stats[0]
                nested_loop_executions = self._planned_nested_loop_executions(plan, index_keys)
            else:
                stats_after = {}
                nested_loop_executions = {}
            index_cardinality_ranges = self._compute_index_stat_bounds(stats_before, stats_after, index_to_outer_query, nested_loop_executions)
        else:
            index_cardinality_ranges = {}

        if result is None:
            assert self.timeout is not None
            execution_data = ExecutionData(query, query_name, plan, None, None, None, None, self.timeout, {}, {}, None)
            executed_as_intended = None
        else:
            explain_json = result[0]
            if not self.verify_plan or plan is None:
                executed_as_intended = None
            else:
                executed_as_intended = self._verify_plan(query, plan, explain_json)
                if executed_as_intended is False:
                    logger = logging.getLogger("Execution Errors")
                    logger.info("Execution plan did not match the expected plan.")
                    logger.info("Time: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    logger.info("Execution query: %s", self.execution_query(query_text, plan, explain_string))
                    if plan is None:
                        logger.info("Expected plan: None")
                    else:
                        logger.info("Expected plan: %s", plan.string())
                    logger.info("Actual plan: %s", explain_json["Plan"])
                    logger.info("")
            execution_data = self._extract_execution_data(query, query_name, plan, explain_json, executed_as_intended, analyze, collect_cardinality_estimates)
        # Merge index stat bounds into explain-derived cardinality ranges
        if executed_as_intended or execution_data.timeout is not None:
            for subquery, idx_range in index_cardinality_ranges.items():
                if subquery in execution_data.cardinality_ranges:
                    existing_range, existing_true = execution_data.cardinality_ranges[subquery]
                    try:
                        merged = existing_range.merge(idx_range)
                    except:
                        merged = idx_range
                    execution_data.cardinality_ranges[subquery] = (merged, existing_true)
                else:
                    execution_data.cardinality_ranges[subquery] = (idx_range, True)
        return execution_data

    def _compute_index_stat_bounds(self, stats_before: Dict[Tuple[Table, Index], int], stats_after: Dict[Tuple[Table, Index], int], index_to_outer_query: Dict[Tuple[Table, Index], Query], nested_loop_executions: Dict[Tuple[Table, Index], int]) -> Dict[Query, CardinalityRange]:
        """Compute cardinality lower bounds from pg_stat_all_indexes deltas. The inner index is
        probed once per outer row per execution of the nested loop, so the delta is divided by
        the number of executions; indexes without a known execution count yield no bound."""
        bounds: Dict[Query, CardinalityRange] = {}
        for index_key, outer_query in index_to_outer_query.items():
            if index_key not in stats_before or index_key not in stats_after or index_key not in nested_loop_executions:
                continue
            executions = nested_loop_executions[index_key]
            if executions <= 0:
                continue
            delta = (stats_after[index_key] - stats_before[index_key]) // executions
            if delta > 0:
                bounds[outer_query] = CardinalityRange(delta, None)
        return bounds

    @staticmethod
    def _executed_nested_loop_executions(explain_plan: dict, index_keys: Set[Tuple[Table, Index]]) -> Dict[Tuple[Table, Index], int]:
        """Actual Loops of the nearest Nested Loop above each index's scan node in the executed
        plan, i.e. how often that nested loop ran (rescans and parallel processes included)."""
        key_lookup = {(table.name(), index.name()): (table, index) for table, index in index_keys}
        executions: Dict[Tuple[Table, Index], int] = {}

        def walk(node: dict, nested_loop_loops: Optional[int]):
            if node["Node Type"] == "Nested Loop":
                nested_loop_loops = node["Actual Loops"]
            if node["Node Type"] in ["Index Scan", "Index Only Scan"] and nested_loop_loops is not None:
                key = key_lookup.get((node.get("Relation Name"), node.get("Index Name")))
                if key is not None:
                    executions[key] = nested_loop_loops
            for child in node.get("Plans", []):
                walk(child, nested_loop_loops)

        walk(explain_plan, None)
        return executions

    def _planned_nested_loop_executions(self, plan: RelationalAlgebraExpression, index_keys: Set[Tuple[Table, Index]]) -> Dict[Tuple[Table, Index], int]:
        """Upper bound on how often each index's nested loop runs, from the plan alone (no
        EXPLAIN after a timeout). PostgreSQL only builds partial join paths whose outer child is
        the partial one, so a nested loop on the outer spine runs once, one below the inner side
        of a hash or merge join runs once per parallel process, and one below the inner side of
        another nested loop is rescanned an unknown number of times and yields no bound."""
        executions: Dict[Tuple[Table, Index], int] = {}

        def walk(node: RelationalAlgebraExpression, inner_of_join: bool, inner_of_nested_loop: bool):
            if isinstance(node, NestedLoopJoinExpression):
                inner = node.inner
                while not isinstance(inner, (ScanExpression, JoinExpression)) and len(inner.children) == 1:
                    inner = inner.children[0]
                if (isinstance(inner, IndexScanExpression) or isinstance(inner, IndexOnlyScanExpression)) and inner.index is not None:
                    key = (inner.table_occurrence.table(), inner.index)
                    if key in index_keys and not inner_of_nested_loop:
                        executions[key] = self.max_parallel_workers_per_gather + 1 if inner_of_join else 1
            if isinstance(node, JoinExpression):
                walk(node.outer, inner_of_join, inner_of_nested_loop)
                walk(node.inner, True, inner_of_nested_loop or isinstance(node, NestedLoopJoinExpression))
            else:
                for child in node.children:
                    walk(child, inner_of_join, inner_of_nested_loop)

        walk(plan, False, False)
        return executions

    def _set(self, cursor):
        for set_command in self.set_commands:
            cursor.execute(set_command)
        if self.timeout is not None:
            cursor.execute("SET statement_timeout = %d;" % self.timeout)

    def _execute_query(self, cursor, query: str) -> Optional[dict]:
        try:
            cursor.execute(query)
        except Exception as e:
            connection = cursor.connection
            cursor.close()
            connection.rollback()
            return None
        explain_json = cursor.fetchone()[0]
        cursor.close()
        return explain_json

    @abstractmethod
    def execution_query(self, query_text: str, plan: RelationalAlgebraExpression, explain_string: str) -> str:
        pass

    @staticmethod
    def _verify_plan(query: Query, plan: RelationalAlgebraExpression, explain_json: dict) -> bool:
        return ExecutionEngine._verify_plan_recursive(query, plan, explain_json["Plan"])

    @staticmethod
    def _verify_plan_recursive(query: Query, plan: RelationalAlgebraExpression, explain_json: dict) -> bool:
        if explain_json["Node Type"] in ExecutionEngine.PASSTHROUGH_NODE_TYPES:
            assert len(explain_json["Plans"]) == 1
            return ExecutionEngine._verify_plan_recursive(query, plan, explain_json["Plans"][0])
        if "Plans" in explain_json:
            if len(plan.children) != len(explain_json["Plans"]):
                return False
            for child_plan, child_explain in zip(plan.children, explain_json["Plans"]):
                if not ExecutionEngine._verify_plan_recursive(query, child_plan, child_explain):
                    return False
        if isinstance(plan, ScanExpression):
            if isinstance(plan, SequentialScanExpression):
                if explain_json["Node Type"] != "Seq Scan":
                    return False
            elif isinstance(plan, IndexScanExpression):
                if explain_json["Node Type"] not in ["Index Scan", "Index Only Scan"]:
                    return False
            elif isinstance(plan, BitmapIndexScanExpression):
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif isinstance(plan, JoinExpression):
            if isinstance(plan, NestedLoopJoinExpression):
                if explain_json["Node Type"] != "Nested Loop":
                    return False
            elif isinstance(plan, HashJoinExpression):
                if explain_json["Node Type"] != "Hash Join":
                    return False
            elif isinstance(plan, MergeJoinExpression):
                if explain_json["Node Type"] != "Merge Join":
                    return False
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return True

    def _extract_execution_data(self, query: Query, query_name: Optional[str], plan: Optional[RelationalAlgebraExpression], explain_json: dict, executed_as_intended: Optional[bool], analyze: bool, collect_cardinality_estimates: bool) -> ExecutionData:
        if isinstance(query, SPJQuery) and analyze:
            safe_cardinalities, risky_cardinalities, _, _ = self._extract_cardinalities_recursive(query, explain_json["Plan"], False)
            cardinalities = {}
            cardinalities.update(safe_cardinalities)
            cardinalities.update(risky_cardinalities)
            if self.verify_cardinalities:
                self._verify_cardinalities(cardinalities)
        else:
            cardinalities = {}
        if collect_cardinality_estimates:
            cardinality_estimates, _ = self._extract_cardinality_estimates_recursive(query, explain_json["Plan"])
        else:
            cardinality_estimates = {}
        return ExecutionData(query,
                             query_name,
                             plan,
                             executed_as_intended,
                             explain_json["Plan"]["Total Cost"],
                             explain_json["Execution Time"] if analyze else None, explain_json["Planning Time"] if analyze else None,
                             None,
                             cardinalities,
                             cardinality_estimates,
                             explain_json=explain_json)


    def _extract_cardinalities_recursive(self,
                                         query: SPJQuery,
                                         explain_json: dict,
                                         underestimates: bool,
                                         outer_table_occurrences: Optional[List[TableOccurrence]] = None,
                                         outer_cardinality_range: Optional[CardinalityRange] = None,
                                         rows_removed_by_join_filter: int = 0,
                                         join_filter_present: bool = False,
                                         gather_rescans: int = 1) -> Tuple[Dict[Query, Tuple[CardinalityRange, bool]], Dict[Query, CardinalityRange], List[TableOccurrence], Optional[CardinalityRange]]:
        if explain_json["Node Type"] in ExecutionEngine.PASSTHROUGH_NODE_TYPES:
            assert len(explain_json["Plans"]) == 1
            child_json = explain_json["Plans"][0]
            underestimates = (explain_json["Node Type"] in ["Materialize", "Memoize"] and child_json["Node Type"] in ["Index Scan", "Index Only Scan"]) or explain_json["Node Type"] == "Gather Merge" or (explain_json["Node Type"] == "Materialize" and underestimates)
            if explain_json["Node Type"] == "Materialize":
                outer_table_occurrences = None
            if explain_json["Node Type"] in ["Gather", "Gather Merge"]:
                # A rescanned Gather (inner of a nested loop) re-executes its whole subtree
                # per rescan, so descendant Actual Loops accumulate rescans x processes and
                # loop-based totals below must be divided by the rescan count.
                gather_rescans = max(1, explain_json["Actual Loops"])
            safe_cardinalities, child_risky_cardinalities, table_occurrences, cardinality_range = self._extract_cardinalities_recursive(query,
                                                                                                                                        child_json,
                                                                                                                                        underestimates,
                                                                                                                                        outer_table_occurrences=outer_table_occurrences,
                                                                                                                                        rows_removed_by_join_filter=rows_removed_by_join_filter,
                                                                                                                                        join_filter_present=join_filter_present,
                                                                                                                                        gather_rescans=gather_rescans)
            risky_cardinalities = {}
            for subplan_query, (subplan_cardinality_range, true_subquery) in child_risky_cardinalities.items():
                if subplan_cardinality_range.min_cardinality > 0 and explain_json["Node Type"] not in ["Gather", "Gather Merge", "Memoize"]:
                    safe_cardinalities[subplan_query] = (subplan_cardinality_range, true_subquery)
                else:
                    risky_cardinalities[subplan_query] = (subplan_cardinality_range, true_subquery)
            return safe_cardinalities, risky_cardinalities, table_occurrences, cardinality_range
        elif explain_json["Node Type"] in ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"]:
            table_occurrence = query.table_occurrence_by_alias(explain_json["Alias"])
            table_occurrences = [table_occurrence]
            subplan_query = SPJQuery(table_occurrences, [], [])
            safe_cardinality_ranges = {}
            risky_cardinality_ranges = {}
            table_cardinality = table_occurrence.table().cardinality()
            if isinstance(table_occurrence.predicate(), TruePredicate):
                subplan_cardinality_range = CardinalityRange(table_cardinality, table_cardinality)
                safe_cardinality_ranges[subplan_query] = (subplan_cardinality_range, True)
            elif explain_json["Node Type"] == "Seq Scan":
                subplan_cardinality_range, _ = self._get_cardinality(explain_json, underestimates, gather_rescans=gather_rescans)
                if subplan_cardinality_range.max_cardinality is None:
                    subplan_cardinality_range = CardinalityRange(subplan_cardinality_range.min_cardinality, table_cardinality)
                    safe_cardinality_ranges[subplan_query] = (subplan_cardinality_range, True)
                else:
                    subplan_cardinality_range = CardinalityRange(subplan_cardinality_range.min_cardinality, min(subplan_cardinality_range.max_cardinality, table_cardinality))
                    risky_cardinality_ranges[subplan_query] = (subplan_cardinality_range, True)
            elif explain_json["Node Type"] in ["Index Scan", "Index Only Scan", "Bitmap Heap Scan"]:
                index_cond = explain_json.get("Index Cond", explain_json.get("Recheck Cond", None))
                is_join_index_scan = index_cond is not None and index_cond.count(".") >= 2
                actual_rows = explain_json["Actual Rows"]
                if is_join_index_scan:
                    actual_loops_per_execution = explain_json["Actual Loops"] / gather_rescans
                    if outer_cardinality_range is None:
                        min_actual_loops = actual_loops_per_execution
                        max_actual_loops = actual_loops_per_execution
                        outer_loops = self.max_parallel_workers_per_gather + 1
                    else:
                        min_actual_loops = outer_cardinality_range.min_cardinality
                        max_actual_loops = outer_cardinality_range.max_cardinality
                        if explain_json["Actual Loops"] == 0:
                            outer_loops = 0
                        else:
                            outer_loops = actual_loops_per_execution / max(min_actual_loops, 1)
                    if outer_table_occurrences is not None:
                        outer_query = query.induced_subquery(outer_table_occurrences)
                        join_graph_edge_dict = query.join_graph_edge_dict()
                        possible_columns = set()
                        for other_table_occurrence, join in join_graph_edge_dict[table_occurrence]:
                            if other_table_occurrence in outer_table_occurrences:
                                for ec_table_occurrence, ec_column in join.equivalence_class():
                                    if ec_table_occurrence == other_table_occurrence:
                                        possible_columns.add((ec_table_occurrence, ec_column))
                        if len(possible_columns) == 1:
                            outer_join_table_occurrence, outer_join_column = possible_columns.pop()
                            outer_join_column_is_unique = outer_query.columns_are_unique(outer_join_table_occurrence, [outer_join_column])
                        else:
                            outer_join_column_is_unique = False
                    else:
                        outer_join_column_is_unique = False
                    if outer_loops > 1:
                        row_uncertainty = 0.01 * (outer_loops + 1)
                    else:
                        row_uncertainty = 0.01
                    if outer_join_column_is_unique and rows_removed_by_join_filter == 0:
                        min_actual_rows = max(0, actual_rows - row_uncertainty)
                        subplan_cardinality_range = CardinalityRange(math.ceil(min_actual_loops * min_actual_rows), table_cardinality)
                    else:
                        subplan_cardinality_range = CardinalityRange(math.ceil(actual_rows), table_cardinality)
                    safe_cardinality_ranges[subplan_query] = (subplan_cardinality_range, True)
                    # A Join Filter above the scan may contain a join clause that the scan counts do not
                    # reflect, even if it removed no rows.
                    if not isinstance(table_occurrence.predicate(), TruePredicate) and outer_table_occurrences is not None and not join_filter_present and self._index_cond_is_pure_join_equalities(index_cond, table_occurrence.alias(), {outer.alias() for outer in outer_table_occurrences}) and len(self.extract_table_aliases(explain_json.get("Filter", ""))) <= 1:
                        parent_table_occurrences = outer_table_occurrences + [table_occurrence]
                        parent_query = query.induced_subquery(parent_table_occurrences)
                        unfiltered_query = parent_query.remove_predicate(table_occurrence)
                        filtered_rows = explain_json.get("Rows Removed by Filter", 0)
                        min_actual_rows = max(0, actual_rows - row_uncertainty)
                        min_filtered_rows = max(0, filtered_rows - 1)
                        min_unfiltered_cardinality = math.floor(min_actual_loops * (min_actual_rows + min_filtered_rows))
                        min_unfiltered_cardinality = max(min_unfiltered_cardinality, subplan_cardinality_range.min_cardinality)
                        if underestimates or max_actual_loops is None:
                            unfiltered_cardinality_range = CardinalityRange(min_unfiltered_cardinality, None)
                            safe_cardinality_ranges[unfiltered_query] = (unfiltered_cardinality_range, False)
                        else:
                            max_unfiltered_cardinality = math.ceil(max_actual_loops * (actual_rows + filtered_rows + 0.5 + row_uncertainty))
                            unfiltered_cardinality_range = CardinalityRange(min_unfiltered_cardinality, max_unfiltered_cardinality)
                            risky_cardinality_ranges[unfiltered_query] = (unfiltered_cardinality_range, False)
                else:
                    subplan_cardinality_range, use_parallel = self._get_cardinality(explain_json, underestimates, gather_rescans=gather_rescans)
                    safe_cardinality_ranges[subplan_query] = (subplan_cardinality_range, True)
                    if index_cond is not None and "Filter" in explain_json:
                        table = table_occurrence.table()
                        unfiltered_table_occurrence = TableOccurrence(table, table_occurrence.alias() + "_unfiltered")
                        parsed_index_condition = parse_table_predicate(unfiltered_table_occurrence, index_cond)
                        unfiltered_table_occurrence.set_predicate(parsed_index_condition)
                        unfiltered_query = SPJQuery([unfiltered_table_occurrence], [], [])
                        actual_loops_per_execution = explain_json["Actual Loops"] / gather_rescans
                        if use_parallel is None:
                            min_actual_loops = 1
                            max_actual_loops = actual_loops_per_execution
                        elif use_parallel:
                            min_actual_loops = actual_loops_per_execution
                            max_actual_loops = actual_loops_per_execution
                        else:
                            min_actual_loops = 1
                            max_actual_loops = 1
                        actual_filtered_rows = explain_json.get("Rows Removed by Filter", 0)
                        filtered_row_uncertainty = 1
                        min_actual_filtered_rows = max(0, actual_filtered_rows - filtered_row_uncertainty)
                        min_filtered_rows = math.floor(min_actual_filtered_rows * min_actual_loops)
                        min_unfiltered_rows = subplan_cardinality_range.min_cardinality + min_filtered_rows
                        if underestimates or subplan_cardinality_range.max_cardinality is None:
                            unfiltered_cardinality_range = CardinalityRange(min_unfiltered_rows, table_cardinality)
                            safe_cardinality_ranges[unfiltered_query] = (unfiltered_cardinality_range, False)
                        else:
                            max_actual_filtered_rows = actual_filtered_rows + filtered_row_uncertainty
                            max_filtered_rows = math.ceil(max_actual_filtered_rows * max_actual_loops)
                            max_unfiltered_rows = subplan_cardinality_range.max_cardinality + max_filtered_rows
                            unfiltered_cardinality_range = CardinalityRange(min_unfiltered_rows, min(max_unfiltered_rows, table_cardinality))
                            risky_cardinality_ranges[unfiltered_query] = (unfiltered_cardinality_range, False)
            else:
                raise NotImplementedError()
            return safe_cardinality_ranges, risky_cardinality_ranges, table_occurrences, subplan_cardinality_range
        elif explain_json["Node Type"] in ["Nested Loop", "Hash Join", "Merge Join"]:
            assert len(explain_json["Plans"]) == 2
            first_child_explain = explain_json["Plans"][0]
            second_child_explain = explain_json["Plans"][1]
            if first_child_explain["Parent Relationship"] == "Outer" and second_child_explain["Parent Relationship"] == "Inner":
                outer_child_explain = first_child_explain
                inner_child_explain = second_child_explain
            elif first_child_explain["Parent Relationship"] == "Inner" and second_child_explain["Parent Relationship"] == "Outer":
                outer_child_explain = second_child_explain
                inner_child_explain = first_child_explain
            else:
                raise ValueError("Unexpected parent relationship in join node: %s" % explain_json["Node Type"])
            is_merge_join = explain_json["Node Type"] == "Merge Join"
            safe_outer_cardinalities, risky_outer_cardinalities, outer_table_occurrences, outer_cardinality_range = self._extract_cardinalities_recursive(query,
                                                                                                                                                          outer_child_explain,
                                                                                                                                                          is_merge_join or underestimates,
                                                                                                                                                          gather_rescans=gather_rescans)

            rows_removed_by_join_filter = explain_json.get("Rows Removed by Join Filter", 0)
            safe_inner_cardinalities, risky_inner_cardinalities, inner_table_occurrences, inner_cardinality_range = self._extract_cardinalities_recursive(query,
                                                                                                                                                          inner_child_explain,
                                                                                                                                                          is_merge_join or (explain_json["Node Type"] == "Nested Loop" and explain_json["Inner Unique"]),
                                                                                                                                                          outer_table_occurrences=outer_table_occurrences,
                                                                                                                                                          outer_cardinality_range=outer_cardinality_range,
                                                                                                                                                          rows_removed_by_join_filter=rows_removed_by_join_filter,
                                                                                                                                                          join_filter_present="Join Filter" in explain_json,
                                                                                                                                                          gather_rescans=gather_rescans)
            cardinality_range, _ = self._get_cardinality(explain_json, underestimates, gather_rescans=gather_rescans)
            safe_cardinality_ranges = {}
            safe_cardinality_ranges.update(safe_outer_cardinalities)
            safe_cardinality_ranges.update(safe_inner_cardinalities)
            risky_cardinality_ranges = {}
            if inner_cardinality_range.min_cardinality > 0 or cardinality_range.min_cardinality > 0:
                risky_cardinality_ranges.update(risky_outer_cardinalities)
            else:
                for subplan_query, (subplan_cardinality_range, true_subquery) in risky_outer_cardinalities.items():
                    if subplan_cardinality_range.min_cardinality > 0:
                        safe_cardinality_ranges[subplan_query] = (CardinalityRange(subplan_cardinality_range.min_cardinality, None), true_subquery)
            if outer_cardinality_range.min_cardinality > 0 or cardinality_range.min_cardinality > 0:
                risky_cardinality_ranges.update(risky_inner_cardinalities)
            else:
                for subplan_query, (subplan_cardinality_range, true_subquery) in risky_inner_cardinalities.items():
                    if subplan_cardinality_range.min_cardinality > 0:
                        safe_cardinality_ranges[subplan_query] = (CardinalityRange(subplan_cardinality_range.min_cardinality, None), true_subquery)
            table_occurrences = outer_table_occurrences + inner_table_occurrences
            subplan_query = query.induced_subquery(table_occurrences)
            risky_cardinality_ranges[subplan_query] = (cardinality_range, True)
            return safe_cardinality_ranges, risky_cardinality_ranges, table_occurrences, cardinality_range
        else:
            raise NotImplementedError(f"Node type {explain_json['Node Type']} not implemented for cardinality extraction.")

    def _get_cardinality(self, explain_json: dict, underestimates: bool, gather_rescans: int = 1) -> Tuple[CardinalityRange, Optional[bool]]:
        # Actual Loops accumulates rescans x processes below a rescanned Gather, and each
        # rescan re-produces the same rows, so loop totals count them gather_rescans times.
        actual_loops_per_execution = explain_json["Actual Loops"] / gather_rescans
        max_cardinality = None
        if "Workers" in explain_json:
            if len(explain_json["Workers"]) == 0:
                use_parallel = False
            else:
                subplan_cardinality = explain_json["Actual Rows"]
                worker_cardinalities = [worker["Actual Rows"] for worker in explain_json["Workers"] if "Actual Rows" in worker]
                if len(worker_cardinalities) == 0:
                    use_parallel = False
                elif all(cardinality == subplan_cardinality for cardinality in worker_cardinalities) and not explain_json["Parallel Aware"]:
                    #  TODO: PostgreSQL is unreliable in this case
                    #  sometimes it is reporting total cardinality and sometimes per-worker cardinality and I am unable to determine which is which
                    #  so "for now" we use the two cases as lower and upper bounds
                    if not underestimates:
                        max_cardinality = explain_json["Actual Rows"] * actual_loops_per_execution + self._parallelism_uncertainty
                    return CardinalityRange(explain_json["Actual Rows"] / gather_rescans, max_cardinality), None
                else:
                    use_parallel = True
        else:
            use_parallel = False
        if use_parallel:
            cardinality = explain_json["Actual Rows"] * actual_loops_per_execution
            if not underestimates:
                max_cardinality = cardinality + self._parallelism_uncertainty
            cardinality_range = CardinalityRange(max(0, cardinality - self._parallelism_uncertainty), max_cardinality)
        else:
            cardinality = explain_json["Actual Rows"]
            if underestimates:
                max_cardinality = None
            else:
                max_cardinality = cardinality
            cardinality_range = CardinalityRange(cardinality, max_cardinality)
        return cardinality_range, use_parallel

    def _verify_cardinalities(self, cardinality_ranges: Dict[Query, Tuple[CardinalityRange, bool]]):
        cursor = self.schema.connection().cursor()
        for query, (cardinality_range, _) in cardinality_ranges.items():
            if isinstance(query, SPJQuery):
                cursor.execute(query.get_query_text(forced_select_clause="COUNT(*)"))
                true_cardinality = cursor.fetchone()[0]
                if true_cardinality < cardinality_range.min_cardinality or (cardinality_range.max_cardinality is not None and true_cardinality > cardinality_range.max_cardinality):
                    if cardinality_range.max_cardinality is None:
                        raise ValueError("True cardinality %d does not match cardinality range %d to infinity for query %s" % (true_cardinality, cardinality_range.min_cardinality, query.query_text()))
                    else:
                        raise ValueError("True cardinality %d does not match cardinality range %d to %d for query %s" % (true_cardinality, cardinality_range.min_cardinality, cardinality_range.max_cardinality, query.query_text()))
        cursor.close()

    @staticmethod
    def extract_table_aliases(filter_string: str) -> Set[str]:
        """Extract all table aliases from a filter string like '(q1.score >= 0 AND u1.id = q1.owner_user_id)'."""
        return set(re.findall(r'\b(\w+)\.\w+', filter_string))

    @staticmethod
    def _index_cond_is_pure_join_equalities(index_cond: str, inner_alias: str, outer_aliases: Set[str]) -> bool:
        """True iff every conjunct of the index condition is a column-to-column equality between
        the inner table and one of the given outer tables. Index conditions containing literals
        (constant comparisons pushed into the index condition) are rejected."""
        if "'" in index_cond:
            return False
        stripped = index_cond.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            stripped = stripped[1:-1]
        for conjunct in stripped.split(" AND "):
            match = re.fullmatch(r"\(?\s*(\w+)\.\w+ = (\w+)\.\w+\s*\)?", conjunct.strip())
            if match is None:
                return False
            left_alias, right_alias = match.group(1), match.group(2)
            if left_alias == inner_alias:
                other_alias = right_alias
            elif right_alias == inner_alias:
                other_alias = left_alias
            else:
                return False
            if other_alias not in outer_aliases:
                return False
        return True

    def _extract_cardinality_estimates_recursive(self, query: SPJQuery, explain_json: dict) -> Tuple[Dict[Query, float], List[TableOccurrence]]:
        if explain_json["Node Type"] in ExecutionEngine.PASSTHROUGH_NODE_TYPES:
            assert len(explain_json["Plans"]) == 1
            child_json = explain_json["Plans"][0]
            return self._extract_cardinality_estimates_recursive(query, child_json)
        elif explain_json["Node Type"] in ["Seq Scan", "Index Scan", "Index Only Scan"]:
            table_occurrence = query.table_occurrence_by_alias(explain_json["Alias"])
            table_occurrences = [table_occurrence]
            cardinality_estimates = {}
            if explain_json["Node Type"] == "Seq Scan":
                subplan_query = SPJQuery([table_occurrence], [], [])
                cardinality_estimate = explain_json["Plan Rows"]
                cardinality_estimates[subplan_query] = cardinality_estimate
            return cardinality_estimates, table_occurrences
        elif explain_json["Node Type"] in ["Nested Loop", "Hash Join", "Merge Join"]:
            assert len(explain_json["Plans"]) == 2
            first_child_explain = explain_json["Plans"][0]
            second_child_explain = explain_json["Plans"][1]
            if first_child_explain["Parent Relationship"] == "Outer" and second_child_explain["Parent Relationship"] == "Inner":
                outer_child_explain = first_child_explain
                inner_child_explain = second_child_explain
            elif first_child_explain["Parent Relationship"] == "Inner" and second_child_explain["Parent Relationship"] == "Outer":
                outer_child_explain = second_child_explain
                inner_child_explain = first_child_explain
            else:
                raise ValueError("Unexpected parent relationship in join node: %s" % explain_json["Node Type"])
            outer_cardinalities, outer_table_occurrences = self._extract_cardinality_estimates_recursive(query, outer_child_explain)
            inner_cardinalities, inner_table_occurrences = self._extract_cardinality_estimates_recursive(query, inner_child_explain)
            cardinality_estimates = {}
            cardinality_estimates.update(outer_cardinalities)
            cardinality_estimates.update(inner_cardinalities)
            table_occurrences = outer_table_occurrences + inner_table_occurrences
            subplan_query = query.induced_subquery(table_occurrences)
            cardinality_estimate = explain_json["Plan Rows"]
            cardinality_estimates[subplan_query] = cardinality_estimate
            return cardinality_estimates, table_occurrences
        else:
            raise NotImplementedError(f"Node type {explain_json['Node Type']} not implemented for cardinality extraction.")



