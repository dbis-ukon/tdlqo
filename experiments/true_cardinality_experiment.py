from typing import List

from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.local_cost_model import LocalCostModel
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from execution_engines.execution_engine import ExecutionEngine
from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from experiments.experiment import Experiment
from optimizers.dummy_optimizer import DummyOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from optimizers.top_down_offline_cost_based_optimizer import TopDownOfflineCostBasedOptimizer
from queries.benchmark_query import BenchmarkQuery
from queries.spj_query import SPJQuery
from schemas.postgresql_configuration import PostgreSQLConfiguration
from schemas.schema import Schema


class TrueCardinalityExperiment(Experiment):
    def __init__(self,
                 schema: Schema,
                 execution_engine: ExecutionEngine,
                 postgresql_configurations: List[PostgreSQLConfiguration],
                 test_set: List[BenchmarkQuery],
                 repetitions: int,
                 gather_on_demand: bool):
        super().__init__("True Cardinality Experiment", "")
        self.schema = schema
        self.execution_engine = execution_engine
        self.postgresql_configurations = postgresql_configurations
        self.test_set = test_set
        self.repetitions = repetitions
        self.gather_on_demand = gather_on_demand

    def run(self):
        cardinality_estimator = PropheticCardinalityEstimator(gather_on_demand_schema=self.schema)
        cost_model = PostgreSQLCostModel(cardinality_estimator, self.schema.postgresql_configuration(), default_width=8, improved=True)
        top_down_learned_optimizer = TopDownLearnedOptimizer(self.schema, TopDownLearnedOptimizerConfiguration.default_configuration(self.schema))
        top_down_learned_optimizer.cost_model = cost_model
        top_down_learned_optimizer.explore([benchmark_query.query for benchmark_query in self.test_set], None, logger=self._logger)
        if self.gather_on_demand:
            cardinality_estimator = cost_model.cardinality_estimator
            assert isinstance(cardinality_estimator, PropheticCardinalityEstimator)
            cardinality_estimator.gather_on_demand_schema = self.schema

        for i, postgresql_configuration in enumerate(self.postgresql_configurations):
            cost_model = PostgreSQLCostModel(cardinality_estimator, postgresql_configuration, improved=True)
            cost_based_optimizer = TopDownOfflineCostBasedOptimizer(self.schema, cost_model, top_down_learned_optimizer.available_operators)
            test_path = self.result_path() + "/cost_based_test_set_%d.csv" % i
            test_file = open(test_path, "w")
            self._logger.info("Running cost-based test set with %d queries for configuration %d" % (len(self.test_set), i))
            hint_path = self.result_path() + "/cost_based_test_set_hints_%d.sql" % i
            hint_file = open(hint_path, "w")
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
            hint_file.close()
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


