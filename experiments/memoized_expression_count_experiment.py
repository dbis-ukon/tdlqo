import csv
import random
import statistics
from typing import Dict, List, Tuple

from experiments.experiment import Experiment
from optimizers.top_down_optimizer import TopDownOptimizer
from queries.benchmark_query import BenchmarkQuery
from queries.spj_query import SPJQuery
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression


class MemoizedExpressionCountExperiment(Experiment):
    """Counts the memoized expressions a top-down optimizer enumerates in every group it visits.

    Each benchmark query is optimized several times and every visited group (a call of the optimizer's
    enumerate_memoized_expressions) is recorded with the number of memoized expressions enumerated for
    it. No query is executed."""

    def __init__(self,
                 optimizer: TopDownOptimizer,
                 benchmark_queries: List[BenchmarkQuery],
                 repetitions: int,
                 seed: int = 0):
        super().__init__("Memoized Expression Count Experiment", "")
        self.optimizer = optimizer
        self.benchmark_queries = benchmark_queries
        self.repetitions = repetitions
        self.seed = seed

    def run(self):
        # The random optimizer draws from the global random module.
        random.seed(self.seed)
        self._logger.info("Optimizing %d queries %d times each with %s" % (len(self.benchmark_queries), self.repetitions, self.optimizer.name))

        group_file = open(self.result_path() + "/groups.csv", "w", newline="")
        group_writer = csv.writer(group_file)
        group_writer.writerow(["query_name", "query_id", "repetition", "visit_index", "group", "num_tables",
                               "forced_index_scan", "num_memoized_expressions"])
        query_file = open(self.result_path() + "/queries.csv", "w", newline="")
        query_writer = csv.writer(query_file)
        query_writer.writerow(["query_name", "query_id", "num_tables", "repetition", "num_groups", "num_memoized_expressions"])

        # Memoized expression counts of all visited groups, by the number of tables in the group.
        counts_by_num_tables: Dict[int, List[int]] = {}
        totals = []
        for benchmark_query in self.benchmark_queries:
            query = benchmark_query.query
            assert isinstance(query, SPJQuery)
            query_num_tables = len(query.table_occurrences())
            query_totals = []
            for repetition in range(self.repetitions):
                visits = self._optimize_and_record(benchmark_query)
                total = 0
                for visit_index, (group, num_memoized_expressions) in enumerate(visits):
                    group_query = group.query()
                    assert isinstance(group_query, SPJQuery)
                    num_tables = len(group_query.table_occurrences())
                    forced_index_scan = group.requirements.force_index_scan is not None
                    group_writer.writerow([benchmark_query.query_name(), benchmark_query.query_id, repetition, visit_index,
                                           group.string(), num_tables, int(forced_index_scan), num_memoized_expressions])
                    if num_tables not in counts_by_num_tables:
                        counts_by_num_tables[num_tables] = []
                    counts_by_num_tables[num_tables].append(num_memoized_expressions)
                    total += num_memoized_expressions
                query_writer.writerow([benchmark_query.query_name(), benchmark_query.query_id, query_num_tables, repetition, len(visits), total])
                query_totals.append(total)
            totals += query_totals
            self._logger.info("%s (%d tables): %.1f memoized expressions per optimization (min %d, max %d)"
                              % (benchmark_query.query_name(), query_num_tables, statistics.mean(query_totals), min(query_totals), max(query_totals)))
            group_file.flush()
            query_file.flush()
        group_file.close()
        query_file.close()

        with open(self.result_path() + "/group_sizes.csv", "w", newline="") as group_size_file:
            group_size_writer = csv.writer(group_size_file)
            group_size_writer.writerow(["num_tables", "num_visits", "mean_memoized_expressions", "median_memoized_expressions",
                                        "min_memoized_expressions", "max_memoized_expressions"])
            for num_tables in sorted(counts_by_num_tables):
                counts = counts_by_num_tables[num_tables]
                group_size_writer.writerow([num_tables, len(counts), "%.3f" % statistics.mean(counts), statistics.median(counts), min(counts), max(counts)])

        group_counts = [count for counts in counts_by_num_tables.values() for count in counts]
        self._logger.info("")
        self._logger.info("Visited groups: %d" % len(group_counts))
        self._logger.info("Mean memoized expressions per group: %.1f" % statistics.mean(group_counts))
        self._logger.info("Median memoized expressions per group: %.1f" % statistics.median(group_counts))
        self._logger.info("Max memoized expressions per group: %d" % max(group_counts))
        self._logger.info("")
        self._logger.info("Optimizations: %d" % len(totals))
        self._logger.info("Mean memoized expressions per optimization: %.1f" % statistics.mean(totals))
        self._logger.info("Median memoized expressions per optimization: %.1f" % statistics.median(totals))
        self._logger.info("Max memoized expressions per optimization: %d" % max(totals))
        self._logger.info("")

    def _optimize_and_record(self, benchmark_query: BenchmarkQuery) -> List[Tuple[GroupRelationalAlgebraExpression, int]]:
        visits = []
        enumerate_memoized_expressions = self.optimizer.enumerate_memoized_expressions

        def recording_enumerate_memoized_expressions(group: GroupRelationalAlgebraExpression):
            memoized_expressions = enumerate_memoized_expressions(group)
            visits.append((group, len(memoized_expressions)))
            return memoized_expressions

        self.optimizer.enumerate_memoized_expressions = recording_enumerate_memoized_expressions
        try:
            self.optimizer.optimize(benchmark_query.query, benchmark_query=benchmark_query)
        finally:
            del self.optimizer.enumerate_memoized_expressions
        return visits
