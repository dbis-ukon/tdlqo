from collections import defaultdict
from typing import List

import numpy as np

from cardinality_estimators.cardinality_estimator import CardinalityMode
from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from execution_engines.execution_engine import ExecutionEngine
from experiments.experiment import Experiment
from optimizers.dummy_optimizer import DummyOptimizer
from optimizers.top_down_cost_based_optimizer import TopDownCostBasedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import TopDownLearnedOptimizerConfiguration
from queries.benchmark_query import BenchmarkQuery
from queries.spj_query import SPJQuery
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from schemas.schema import Schema


def _node_label(node):
    cls = type(node).__name__.replace("Expression", "")
    if isinstance(node, ScanExpression):
        to = node.table_occurrence
        alias = to.alias() if to.alias() is not None else to.table().name()
        extra = ""
        if isinstance(node, IndexScanExpression):
            idx_name = node.index.name() if node.index is not None else "None"
            extra = f" idx={idx_name} memoized={node.memoized} join_clause={node.join_clause}"
        return f"{cls}({to.table().name()} AS {alias}){extra}"
    return cls


def _collect_local_costs(node, cost_model, acc):
    acc[id(node)] = cost_model.local_cost(node, cardinality_mode=CardinalityMode.MEAN)
    for child in node.children:
        _collect_local_costs(child, cost_model, acc)


def _cumulative_from_local(node, local_by_id):
    local = local_by_id[id(node)]
    if local is None:
        return None
    total = local
    for child in node.children:
        child_total = _cumulative_from_local(child, local_by_id)
        if child_total is None:
            return None
        total += child_total
    return total


def _dump_plan(f, node, cost_model, cardinality_estimator, depth=0, local_by_id=None):
    if local_by_id is None:
        local_by_id = {}
        _collect_local_costs(node, cost_model, local_by_id)
    local = local_by_id[id(node)]
    cumulative = _cumulative_from_local(node, local_by_id)
    card = cardinality_estimator.estimate(node.canonical_query(), cardinality_mode=CardinalityMode.MEAN)
    card_str = f"{card:.0f}" if card is not None else "None"
    local_str = f"{local:.2f}" if local is not None else "None"
    cum_str = f"{cumulative:.2f}" if cumulative is not None else "None"
    indent = "  " * depth
    f.write(f"{indent}- {_node_label(node)} local={local_str} cum={cum_str} card={card_str}\n")
    for child in node.children:
        _dump_plan(f, child, cost_model, cardinality_estimator, depth=depth + 1, local_by_id=local_by_id)


class PostgreSQLCardinalityExperiment(Experiment):
    def __init__(self,
                 schema: Schema,
                 execution_engine: ExecutionEngine,
                 test_set: List[BenchmarkQuery],
                 repetitions: int):
        super().__init__("PostgreSQL Cardinality Experiment", "")
        self.schema = schema
        self.execution_engine = execution_engine
        self.test_set = test_set
        self.repetitions = repetitions

    def run(self):
        cardinality_estimator = PostgreSQLCardinalityEstimator(self.schema, cache=True)
        cost_model = PostgreSQLCostModel(cardinality_estimator, self.schema.postgresql_configuration(), default_width=8, improved=True, probe_cache_penalty=2)
        cost_based_optimizer = TopDownCostBasedOptimizer(cost_model, TopDownLearnedOptimizerConfiguration.default_configuration(self.schema).available_operators)

        test_path = self.result_path() + "/cost_based_test_set.csv"
        test_file = open(test_path, "w")
        self._logger.info("Running cost-based test set with %d queries" % len(self.test_set))
        hint_path = self.result_path() + "/cost_based_test_set_hints.sql"
        hint_file = open(hint_path, "w")
        details_path = self.result_path() + "/cost_based_plan_details.txt"
        details_file = open(details_path, "w")
        test_end_to_end_data_list = []
        for j in range(self.repetitions):
            for benchmark_query in self.test_set:
                end_to_end_data = self._optimize_execute_memorize(cost_based_optimizer, [], self.execution_engine, benchmark_query, test_file)
                test_end_to_end_data_list.append(end_to_end_data)
                if j == 0:
                    plan = cost_based_optimizer.optimize(benchmark_query.query)
                    execution_query = self.execution_engine.execution_query(benchmark_query.query_text, plan, "EXPLAIN (ANALYZE TRUE, VERBOSE TRUE, FORMAT JSON)")
                    hint_file.write("Query %s\n" % benchmark_query.query_name())
                    hint_file.write(execution_query + "\n")
                    for subplan, cost in cost_model.sublan_costs(plan).items():
                        subplan_query = subplan.query()
                        assert isinstance(subplan_query, SPJQuery)
                        subplan_aliases = ",".join([table_occurrence.alias() for table_occurrence in subplan_query.table_occurrences()])
                        hint_file.write("(%s):  %.4f\n" % (subplan_aliases, cost))
                    hint_file.write("\n\n")

                    details_file.write("Query %s\n" % benchmark_query.query_name())
                    details_file.write(execution_query + "\n")
                    total_cost = cost_model.cost(plan)
                    total_cost_str = "%.2f" % total_cost if total_cost is not None else "None"
                    details_file.write("Total estimated cost: %s\n" % total_cost_str)
                    details_file.write("Plan tree:\n")
                    _dump_plan(details_file, plan, cost_model, cardinality_estimator, depth=1)
                    details_file.write("Subplan costs:\n")
                    for subplan, cost in cost_model.sublan_costs(plan).items():
                        subplan_query = subplan.query()
                        assert isinstance(subplan_query, SPJQuery)
                        subplan_aliases = ",".join([table_occurrence.alias() for table_occurrence in subplan_query.table_occurrences()])
                        details_file.write("  (%s): %.4f\n" % (subplan_aliases, cost))
                    details_file.write("\n\n")
        hint_file.close()
        details_file.close()
        test_file.close()
        self._logger.info("")
        self._logger.info(self._print_end_to_end_stats(test_end_to_end_data_list))

        dummy_optimizer = DummyOptimizer()
        postgresql_test_path = self.result_path() + "/postgresql_test_set.csv"
        postgresql_test_file = open(postgresql_test_path, "w")
        self._logger.info("Running PostgreSQL test set with %d queries" % len(self.test_set))
        postgresql_end_to_end_data_list = []
        for _ in range(self.repetitions):
            for benchmark_query in self.test_set:
                end_to_end_data = self._optimize_execute_memorize(dummy_optimizer, [], self.execution_engine, benchmark_query, postgresql_test_file)
                postgresql_end_to_end_data_list.append(end_to_end_data)
        postgresql_test_file.close()
        self._logger.info("")
        self._logger.info(self._print_end_to_end_stats(postgresql_end_to_end_data_list))

        cost_based_times = defaultdict(list)
        postgresql_times = defaultdict(list)
        for j in range(self.repetitions):
            for i, benchmark_query in enumerate(self.test_set):
                idx = j * len(self.test_set) + i
                cb_data = test_end_to_end_data_list[idx].execution_data
                cb_time = cb_data.execution_time if cb_data.execution_time is not None else cb_data.timeout
                cost_based_times[benchmark_query.query_name()].append(cb_time)
                pg_data = postgresql_end_to_end_data_list[idx].execution_data
                pg_time = pg_data.execution_time if pg_data.execution_time is not None else pg_data.timeout
                postgresql_times[benchmark_query.query_name()].append(pg_time)

        query_stats = []
        for benchmark_query in self.test_set:
            qname = benchmark_query.query_name()
            cb_gmean = float(np.exp(np.mean(np.log(cost_based_times[qname]))))
            pg_gmean = float(np.exp(np.mean(np.log(postgresql_times[qname]))))
            ratio = cb_gmean / pg_gmean
            query_stats.append((qname, cb_gmean, pg_gmean, ratio))

        query_stats.sort(key=lambda x: x[3], reverse=True)

        self._logger.info("")
        self._logger.info("Per-query geometric mean runtimes (sorted by ratio, our slowest first):")
        self._logger.info("%-20s %15s %15s %10s" % ("Query", "CostBased (ms)", "PostgreSQL (ms)", "Ratio"))
        for qname, cb_gmean, pg_gmean, ratio in query_stats:
            self._logger.info("%-20s %15.1f %15.1f %10.2f" % (qname, cb_gmean, pg_gmean, ratio))
