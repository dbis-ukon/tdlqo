import csv
from itertools import cycle, islice
from typing import Callable, Dict, List, Tuple
import json
import logging
import time

from execution_engines.execution_engine import ExecutionEngine
from experiments.experiment import Experiment
from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression


class ConsistencyExperiment(Experiment):
    def __init__(self,
                 optimizer_generator: Callable[[List[BenchmarkQuery], List[BenchmarkQuery]], Optimizer],
                 execution_engine: ExecutionEngine,
                 iterations: int,
                 training_queries: List[BenchmarkQuery],
                 queries_per_iteration: int = 100,
                 debug_mode: bool = False):
        super().__init__("Consistency Experiment", "")
        self.optimizer_generator = optimizer_generator
        self.execution_engine = execution_engine
        self.iterations = iterations
        self.training_queries = training_queries
        self.queries_per_iteration = queries_per_iteration

        if debug_mode:
            debug_logger = logging.getLogger("Optimizer Debug %s" % self.start_time.strftime("%Y%m%d_%H%M%S"))
            debug_logger.setLevel(logging.INFO)
            debug_logger.propagate = False
            debug_handler = logging.FileHandler(self.result_path() + "/optimizer_debug.log")
            debug_handler.setLevel(logging.INFO)
            debug_logger.addHandler(debug_handler)
            self._debug_logger = debug_logger

    def run(self):
        optimizer = self.optimizer_generator(self.training_queries, self.training_queries)
        optimizer_start_time = time.time()
        optimizer.print_parameters(self._logger)
        optimizer.optimizer_id = "0"

        if len(self.training_queries) > self.queries_per_iteration:
            iteration_training_query_list = []
            it = cycle(self.training_queries)
            for _ in range(self.iterations):
                iteration_training_query_list.append(list(islice(it, self.queries_per_iteration)))
        else:
            iteration_training_query_list = [self.training_queries] * self.iterations

        for iteration, iteration_training_queries in enumerate(iteration_training_query_list):
            self._iteration_train(iteration, optimizer, iteration_training_queries)

            training_meta_data = optimizer.train(logger=self._logger)

            current_time = time.time()
            training_meta_data["optimizer_training_time_seconds"] = current_time - optimizer_start_time
            training_meta_data_path = self.result_path() + "/training_meta_data_iteration_0_%d.json" % (iteration + 1)
            with open(training_meta_data_path, "w") as f:
                json.dump(training_meta_data, f, indent=4)
            optimizer_path = self.result_path() + "/optimizer_iteration_0_%d" % (iteration + 1)
            optimizer.save(optimizer_path)

            probe_start_time = time.time()
            self._iteration_consistency(iteration, optimizer, self.training_queries)
            probe_end_time = time.time()
            optimizer_start_time = optimizer_start_time + (probe_end_time - probe_start_time)

    def _iteration_train(self, iteration: int, optimizer: Optimizer, training_queries: List[BenchmarkQuery]):
        training_end_to_end_data_list = []
        training_iteration_path = self.result_path() + "/training_iteration_%d.csv" % iteration
        training_iteration_file = open(training_iteration_path, "a")
        for benchmark_query in training_queries:
            end_to_end_data = self._optimize_execute_memorize(optimizer, [optimizer], self.execution_engine, benchmark_query, training_iteration_file, explore=True)
            training_end_to_end_data_list.append(end_to_end_data)
        training_iteration_file.close()
        self._logger.info("")
        self._logger.info("Training iteration %d completed" % iteration)
        self._logger.info(self._print_end_to_end_stats(training_end_to_end_data_list))
        self._logger.info("")

    def _iteration_consistency(self, iteration: int, optimizer: Optimizer, queries: List[BenchmarkQuery]):
        consistency_path = self.result_path() + "/consistency_iteration_%d.csv" % iteration
        with open(consistency_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_name", "subplan_size", "parent_op", "inherited_sort_req",
                "in_context_strict_sig", "standalone_strict_sig",
                "strict_match", "canonical_match",
            ])
            n_subplans = 0
            n_strict = 0
            n_canonical = 0
            n_no_req = 0
            n_no_req_strict = 0
            for benchmark_query in queries:
                query = benchmark_query.query
                if not isinstance(query, SPJQuery):
                    continue
                try:
                    in_context_plan = optimizer.optimize(query, explore=False, benchmark_query=benchmark_query)
                except Exception as e:
                    self._logger.warning("In-context optimization failed for %s: %s" % (benchmark_query.query_name(), e))
                    continue
                if in_context_plan is None:
                    continue
                parent_map, subtrees = self._collect_join_subtrees(in_context_plan)
                seen_table_sets = set()
                for subtree in subtrees:
                    table_occs = sorted(self._collect_table_occurrences(subtree), key=lambda t: t.sort_key())
                    table_set_key = frozenset(table_occs)
                    if table_set_key in seen_table_sets:
                        continue
                    seen_table_sets.add(table_set_key)

                    parent = parent_map[id(subtree)]
                    parent_op = type(parent).__name__
                    sort_req = self._inherited_sort_req(parent_map, subtree)
                    in_context_strict = self._strict_signature(subtree)
                    in_context_canonical = self._canonical_signature(subtree)

                    induced = query.induced_subquery(list(table_occs))
                    aliases = "_".join(t.alias() for t in table_occs)
                    induced.query_name = "%s__sub__%s" % (benchmark_query.query_name(), aliases)
                    try:
                        standalone_plan = optimizer.optimize(induced, explore=False, benchmark_query=None)
                    except Exception as e:
                        self._logger.warning("Standalone optimization failed for %s subplan {%s}: %s" % (benchmark_query.query_name(), aliases, e))
                        continue
                    if standalone_plan is None:
                        continue
                    standalone_strict = self._strict_signature(standalone_plan)
                    standalone_canonical = self._canonical_signature(standalone_plan)
                    strict_match = in_context_strict == standalone_strict
                    canonical_match = in_context_canonical == standalone_canonical

                    writer.writerow([
                        benchmark_query.query_name(),
                        len(table_occs),
                        parent_op,
                        1 if sort_req else 0,
                        in_context_strict,
                        standalone_strict,
                        1 if strict_match else 0,
                        1 if canonical_match else 0,
                    ])
                    n_subplans += 1
                    if strict_match:
                        n_strict += 1
                    if canonical_match:
                        n_canonical += 1
                    if not sort_req:
                        n_no_req += 1
                        if strict_match:
                            n_no_req_strict += 1

        self._logger.info("")
        self._logger.info("Consistency probe iteration %d completed" % iteration)
        if n_subplans > 0:
            self._logger.info("Subplans probed: %d" % n_subplans)
            self._logger.info("Strict match rate: %d/%d = %.3f" % (n_strict, n_subplans, n_strict / n_subplans))
            self._logger.info("Canonical match rate: %d/%d = %.3f" % (n_canonical, n_subplans, n_canonical / n_subplans))
        if n_no_req > 0:
            self._logger.info("Strict match rate (no inherited sort req): %d/%d = %.3f" % (n_no_req_strict, n_no_req, n_no_req_strict / n_no_req))
        self._logger.info("")

    @staticmethod
    def _collect_join_subtrees(plan: RelationalAlgebraExpression) -> Tuple[Dict[int, RelationalAlgebraExpression], List[RelationalAlgebraExpression]]:
        parent_map: Dict[int, RelationalAlgebraExpression] = {}
        subtrees: List[RelationalAlgebraExpression] = []

        def walk(node: RelationalAlgebraExpression, parent: RelationalAlgebraExpression):
            if parent is not None:
                parent_map[id(node)] = parent
                if isinstance(node, JoinExpression):
                    subtrees.append(node)
            for child in node.children:
                walk(child, node)

        walk(plan, None)
        return parent_map, subtrees

    @staticmethod
    def _collect_table_occurrences(node: RelationalAlgebraExpression) -> List[TableOccurrence]:
        if isinstance(node, ScanExpression):
            return [node.table_occurrence]
        occs: List[TableOccurrence] = []
        for child in node.children:
            occs.extend(ConsistencyExperiment._collect_table_occurrences(child))
        return occs

    @staticmethod
    def _inherited_sort_req(parent_map: Dict[int, RelationalAlgebraExpression], target: RelationalAlgebraExpression) -> bool:
        cur = target
        while id(cur) in parent_map:
            p = parent_map[id(cur)]
            if isinstance(p, MergeJoinExpression):
                return True
            if isinstance(p, HashJoinExpression):
                return False
            if isinstance(p, NestedLoopJoinExpression):
                if cur is p.inner:
                    return False
                cur = p
                continue
            return False
        return False

    @staticmethod
    def _scan_signature(node: ScanExpression) -> str:
        sig = "%s(%s" % (type(node).__name__, node.table_occurrence.alias())
        index = getattr(node, "index", None)
        if index is not None:
            sig += ":" + index.name()
        sig += ")"
        return sig

    @staticmethod
    def _strict_signature(node: RelationalAlgebraExpression) -> str:
        if isinstance(node, ScanExpression):
            return ConsistencyExperiment._scan_signature(node)
        if isinstance(node, JoinExpression):
            return "%s(%s|%s)" % (
                type(node).__name__,
                ConsistencyExperiment._strict_signature(node.outer),
                ConsistencyExperiment._strict_signature(node.inner),
            )
        return "%s(%s)" % (
            type(node).__name__,
            "|".join(ConsistencyExperiment._strict_signature(c) for c in node.children),
        )

    @staticmethod
    def _canonical_signature(node: RelationalAlgebraExpression) -> str:
        if isinstance(node, ScanExpression):
            return ConsistencyExperiment._scan_signature(node)
        if isinstance(node, JoinExpression):
            sigs = sorted([
                ConsistencyExperiment._canonical_signature(node.outer),
                ConsistencyExperiment._canonical_signature(node.inner),
            ])
            return "%s(%s|%s)" % (type(node).__name__, sigs[0], sigs[1])
        child_sigs = sorted(ConsistencyExperiment._canonical_signature(c) for c in node.children)
        return "%s(%s)" % (type(node).__name__, "|".join(child_sigs))
