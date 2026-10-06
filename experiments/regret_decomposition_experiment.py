import csv
import statistics
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

from cost_models.local_cost_model import LocalCostModel
from experiments.experiment import Experiment
from optimizers.top_down_cost_based_optimizer import TopDownCostBasedOptimizer, _IdentityGroup
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from queries.benchmark_query import BenchmarkQuery
from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements
from schemas.schema import Schema

# Join orders are compared as the ordered pair of subplan queries a multiexpression splits its
# group into, so two multiexpressions share a join order exactly when they have the same outer
# and the same inner subplan query, whatever operator and subgroup requirements they use.
JoinOrder = Optional[Tuple[FrozenSet[TableOccurrence], FrozenSet[TableOccurrence]]]

# Tolerance for comparing recomputed costs with the costs of the dynamic program.
_RELATIVE_TOLERANCE = 1e-6


class _GroupAnalysis:
    """What the teacher knows about one group: the cost of every candidate multiexpression
    assuming an optimal continuation below it, and the best cost of each join order."""

    def __init__(self,
                 group: GroupRelationalAlgebraExpression,
                 candidates: List[RelationalAlgebraExpression],
                 optimal_cost: float,
                 candidate_costs: Dict[RelationalAlgebraExpression, Optional[float]],
                 join_orders: Dict[RelationalAlgebraExpression, JoinOrder],
                 join_order_costs: Dict[JoinOrder, float]):
        self.group = group
        self.candidates = candidates
        self.optimal_cost = optimal_cost
        self.candidate_costs = candidate_costs
        self.join_orders = join_orders
        self.join_order_costs = join_order_costs

    def num_tables(self) -> int:
        query = self.group.query()
        assert isinstance(query, SPJQuery)
        return len(query.table_occurrences())


class RegretDecompositionExperiment(Experiment):
    """Decomposes what the learned optimizer gets wrong into a join ordering and a physical
    implementation part, measured by the teacher cost model on true cardinalities.

    For every group in the search space of the test queries, the teacher gives the optimal
    total cost C* and, for each candidate multiexpression, the cost of taking that candidate
    and continuing optimally. The regret of the candidate the learned optimizer picks splits
    into the cost of its join order (the best candidate sharing that join order, against C*)
    and the cost of its physical implementation (the picked candidate against that best one).
    Both are reported relative to C*, and averaged over all groups. The search space depends only
    on the query, so all checkpoints are scored on the same groups."""

    def __init__(self,
                 schema: Schema,
                 configuration: TopDownLearnedOptimizerConfiguration,
                 cost_model: LocalCostModel,
                 splits: List[Tuple[int, List[Tuple[int, str]], List[BenchmarkQuery]]],
                 min_candidates: int = 2):
        super().__init__("Regret Decomposition Experiment", "")
        assert not configuration.double_q_learning, "the scored choice must be deterministic"
        # Each split is scored with its own checkpoints on its own held-out queries.
        iterations = [[iteration for iteration, _ in checkpoints] for _, checkpoints, _ in splits]
        assert all(split_iterations == iterations[0] for split_iterations in iterations), \
            "the splits must share their iterations for the results to be pooled"
        self.schema = schema
        self.configuration = configuration
        self.cost_model = cost_model
        self.splits = splits
        self.checkpoints = splits[0][1]
        # Groups with fewer candidates (e.g. a single forced index scan) hold no decision and are skipped.
        self.min_candidates = min_candidates

    def run(self):
        group_file = open(self.result_path() + "/group_regrets.csv", "w", newline="")
        group_writer = csv.writer(group_file)
        group_writer.writerow(["iteration", "split", "query_name", "query_id", "group", "num_tables",
                               "num_candidates", "num_join_orders", "chosen_operator",
                               "optimal_cost", "chosen_cost", "best_same_join_order_cost",
                               "join_order_regret", "physical_regret",
                               "join_order_optimal", "physical_optimal", "spearman"])

        regrets: Dict[int, List[Tuple[float, float, bool]]] = {iteration: [] for iteration, _ in self.checkpoints}
        infeasible: Dict[int, int] = {iteration: 0 for iteration, _ in self.checkpoints}
        correlations: Dict[int, List[float]] = {iteration: [] for iteration, _ in self.checkpoints}

        for split, checkpoints, test_queries in self.splits:
            optimizers = []
            for iteration, checkpoint_path in checkpoints:
                optimizer = TopDownLearnedOptimizer(self.schema, self.configuration)
                optimizer.load(checkpoint_path)
                optimizers.append((iteration, optimizer))
            self._logger.info("Split %d: loaded %d checkpoints, %d test queries"
                              % (split, len(optimizers), len(test_queries)))
            self._run_split(split, optimizers, test_queries, group_writer, group_file, regrets,
                            infeasible, correlations)
        group_file.close()

        self._write_summary(regrets, infeasible, correlations)

    def _run_split(self, split, optimizers, test_queries, group_writer, group_file,
                   regrets, infeasible, correlations):
        for benchmark_query in test_queries:
            query = benchmark_query.query
            analyses, dynamic_programming_seconds, skipped = self._analyze_query(query)
            self._logger.info("split %d, %s: %d groups scored, %d skipped, teacher dynamic program %.1f s"
                              % (split, benchmark_query.query_name(), len(analyses), skipped, dynamic_programming_seconds))
            for iteration, optimizer in optimizers:
                start_time = time.time()
                for analysis in analyses:
                    predicted_costs = optimizer.get_costs(analysis.group, analysis.candidates)
                    chosen = min(analysis.candidates, key=lambda candidate: predicted_costs[candidate])
                    chosen_cost = analysis.candidate_costs[chosen]
                    if chosen_cost is None:
                        # The learned optimizer picked a candidate the teacher cannot cost, so
                        # there is no finite regret to attribute.
                        infeasible[iteration] += 1
                        continue
                    ranked = [(cost, predicted_costs[candidate])
                              for candidate, cost in analysis.candidate_costs.items() if cost is not None]
                    correlation = self._spearman_correlation([cost for cost, _ in ranked],
                                                             [predicted for _, predicted in ranked])
                    best_same_join_order_cost = analysis.join_order_costs[analysis.join_orders[chosen]]
                    join_order_regret = (best_same_join_order_cost - analysis.optimal_cost) / analysis.optimal_cost
                    physical_regret = (chosen_cost - best_same_join_order_cost) / analysis.optimal_cost
                    assert join_order_regret >= -_RELATIVE_TOLERANCE, join_order_regret
                    assert physical_regret >= -_RELATIVE_TOLERANCE, physical_regret
                    join_order_regret = max(join_order_regret, 0.0)
                    physical_regret = max(physical_regret, 0.0)
                    is_join_group = isinstance(chosen, JoinExpression)
                    regrets[iteration].append((join_order_regret, physical_regret, is_join_group))
                    if correlation is not None:
                        correlations[iteration].append(correlation)
                    group_writer.writerow([iteration, split, benchmark_query.query_name(), benchmark_query.query_id,
                                           analysis.group.string(), analysis.num_tables(),
                                           len(analysis.candidates), len(analysis.join_order_costs),
                                           type(chosen).__name__,
                                           "%.6f" % analysis.optimal_cost, "%.6f" % chosen_cost,
                                           "%.6f" % best_same_join_order_cost,
                                           "%.6f" % join_order_regret, "%.6f" % physical_regret,
                                           int(join_order_regret <= _RELATIVE_TOLERANCE),
                                           int(physical_regret <= _RELATIVE_TOLERANCE),
                                           "" if correlation is None else "%.6f" % correlation])
                self._logger.info("    iteration %d scored in %.1f s" % (iteration, time.time() - start_time))
            group_file.flush()

    def _write_summary(self,
                       regrets: Dict[int, List[Tuple[float, float, bool]]],
                       infeasible: Dict[int, int],
                       correlations: Dict[int, List[float]]):
        summary_path = self.result_path() + "/iteration_summary.csv"
        with open(summary_path, "w", newline="") as summary_file:
            summary_writer = csv.writer(summary_file)
            summary_writer.writerow(["iteration", "num_groups", "num_join_groups", "num_scan_groups",
                                     "num_infeasible", "mean_join_order_regret", "mean_physical_regret",
                                     "mean_total_regret", "median_total_regret",
                                     "join_order_optimal_rate", "physical_optimal_rate",
                                     "mean_join_order_regret_join_groups", "mean_physical_regret_join_groups",
                                     "mean_spearman", "num_ranked_groups"])
            for iteration, _ in self.checkpoints:
                group_regrets = regrets[iteration]
                if not group_regrets:
                    self._logger.info("Iteration %d: no groups scored" % iteration)
                    continue
                join_order_regrets = [join_order_regret for join_order_regret, _, _ in group_regrets]
                physical_regrets = [physical_regret for _, physical_regret, _ in group_regrets]
                totals = [join_order_regret + physical_regret
                          for join_order_regret, physical_regret, _ in group_regrets]
                join_group_regrets = [(join_order_regret, physical_regret)
                                      for join_order_regret, physical_regret, is_join_group in group_regrets
                                      if is_join_group]
                num_join_groups = len(join_group_regrets)
                summary_writer.writerow([
                    iteration, len(group_regrets), num_join_groups, len(group_regrets) - num_join_groups,
                    infeasible[iteration],
                    "%.6f" % statistics.mean(join_order_regrets),
                    "%.6f" % statistics.mean(physical_regrets),
                    "%.6f" % statistics.mean(totals),
                    "%.6f" % statistics.median(totals),
                    "%.6f" % (sum(1 for regret in join_order_regrets if regret <= _RELATIVE_TOLERANCE) / len(group_regrets)),
                    "%.6f" % (sum(1 for regret in physical_regrets if regret <= _RELATIVE_TOLERANCE) / len(group_regrets)),
                    "%.6f" % (statistics.mean([j for j, _ in join_group_regrets]) if num_join_groups else 0.0),
                    "%.6f" % (statistics.mean([p for _, p in join_group_regrets]) if num_join_groups else 0.0),
                    "%.6f" % (statistics.mean(correlations[iteration]) if correlations[iteration] else 0.0),
                    len(correlations[iteration])])
                self._logger.info(
                    "Iteration %2d: mean relative regret %.4f (join order %.4f, physical %.4f), "
                    "mean Spearman %.4f, over %d groups"
                    % (iteration, statistics.mean(totals), statistics.mean(join_order_regrets),
                       statistics.mean(physical_regrets),
                       statistics.mean(correlations[iteration]) if correlations[iteration] else float('nan'),
                       len(group_regrets)))
        self._logger.info("Summary written to %s" % summary_path)

    @staticmethod
    def _spearman_correlation(teacher_costs: List[float], predicted_costs: List[float]) -> Optional[float]:
        """Rank correlation between the teacher's and the model's ordering of one group's
        candidates. Ties share their average rank, so candidates the teacher costs equally
        cannot count as a ranking mistake. None when either side ranks everything equally."""
        if len(teacher_costs) < 2:
            return None

        def average_ranks(values: List[float]) -> List[float]:
            order = sorted(range(len(values)), key=lambda index: values[index])
            ranks = [0.0] * len(values)
            position = 0
            while position < len(order):
                end = position
                while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                    end += 1
                average_rank = (position + end) / 2 + 1
                for tied in range(position, end + 1):
                    ranks[order[tied]] = average_rank
                position = end + 1
            return ranks

        teacher_ranks = average_ranks(teacher_costs)
        predicted_ranks = average_ranks(predicted_costs)
        teacher_mean = statistics.mean(teacher_ranks)
        predicted_mean = statistics.mean(predicted_ranks)
        covariance = sum((t - teacher_mean) * (p - predicted_mean)
                         for t, p in zip(teacher_ranks, predicted_ranks))
        teacher_variance = sum((t - teacher_mean) ** 2 for t in teacher_ranks)
        predicted_variance = sum((p - predicted_mean) ** 2 for p in predicted_ranks)
        if teacher_variance <= 0 or predicted_variance <= 0:
            return None
        return covariance / (teacher_variance * predicted_variance) ** 0.5

    def _analyze_query(self, query: Query) -> Tuple[List[_GroupAnalysis], float, int]:
        cost_based_optimizer = TopDownCostBasedOptimizer(self.cost_model, self.configuration.available_operators)
        start_time = time.time()
        # Optimizing fills the memory with the optimal cost of every reachable group, which is
        # exactly the search space of the query.
        cost_based_optimizer.optimize(query)
        dynamic_programming_seconds = time.time() - start_time

        memory = cost_based_optimizer._optimization_memory
        analyses = []
        skipped = 0
        for identity_group, (_, optimal_cost) in list(memory.items()):
            group = identity_group.group
            if optimal_cost is None or optimal_cost <= 0:
                skipped += 1
                continue
            candidates = cost_based_optimizer.enumerate_memoized_expressions(group)
            if len(candidates) < self.min_candidates:
                skipped += 1
                continue
            candidate_costs: Dict[RelationalAlgebraExpression, Optional[float]] = {}
            join_orders: Dict[RelationalAlgebraExpression, JoinOrder] = {}
            join_order_costs: Dict[JoinOrder, float] = {}
            for candidate in candidates:
                cost = self._candidate_cost(candidate, memory)
                candidate_costs[candidate] = cost
                join_order = self._join_order(candidate)
                join_orders[candidate] = join_order
                if cost is not None and (join_order not in join_order_costs or cost < join_order_costs[join_order]):
                    join_order_costs[join_order] = cost
            feasible_costs = [cost for cost in candidate_costs.values() if cost is not None]
            assert feasible_costs, "a group with an optimal cost has a feasible candidate"
            # Sanity check: the recomputed candidate costs reproduce the optimum of the dynamic program.
            assert abs(min(feasible_costs) - optimal_cost) <= _RELATIVE_TOLERANCE * optimal_cost, \
                "recomputed optimum %f does not match the dynamic program's %f" % (min(feasible_costs), optimal_cost)
            analyses.append(_GroupAnalysis(group, candidates, optimal_cost, candidate_costs, join_orders, join_order_costs))
        return analyses, dynamic_programming_seconds, skipped

    def _candidate_cost(self,
                        candidate: RelationalAlgebraExpression,
                        memory: Dict[_IdentityGroup, Tuple[RelationalAlgebraExpression, Optional[float]]]) -> Optional[float]:
        """The cost of taking this multiexpression and continuing optimally below it."""
        cost = self.cost_model.local_cost(candidate)
        if cost is None:
            return None
        for child_group in candidate.children:
            assert isinstance(child_group, GroupRelationalAlgebraExpression)
            child_cost = memory[_IdentityGroup(child_group)][1]
            if child_cost is None:
                return None
            cost += child_cost
        return cost

    @staticmethod
    def _join_order(candidate: RelationalAlgebraExpression) -> JoinOrder:
        if not isinstance(candidate, JoinExpression):
            # Scans do not split their group, so all of them share one join order.
            return None
        outer_query = candidate.outer.query()
        inner_query = candidate.inner.query()
        assert isinstance(outer_query, SPJQuery) and isinstance(inner_query, SPJQuery)
        return outer_query.table_occurrences(), inner_query.table_occurrences()
