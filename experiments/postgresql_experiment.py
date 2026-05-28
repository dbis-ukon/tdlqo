from typing import List, Optional

import numpy as np

from execution_engines.postgresql_execution_engine import PostgreSQLExecutionEngine
from experiments.experiment import Experiment
from optimizers.dummy_optimizer import DummyOptimizer
from queries.benchmark_query import BenchmarkQuery
from schemas.schema import Schema


class PostgreSQLExperiment(Experiment):
    def __init__(self, schema: Schema, test_sets: List[List[BenchmarkQuery]], repetitions: int = 1, timeout: Optional[int] = None):
        super().__init__("PostgreSQL Experiment", "")
        self.test_sets = test_sets
        self.repetitions = repetitions
        operator_disable_commands = ["SET enable_bitmapscan = off;",
                                     "SET enable_mergejoin = off;"]
        reduce_random_page_cost_commands = ["SET random_page_cost = 1.1;"]
        no_page_cost_commands = ["SET random_page_cost = 0;", "SET seq_page_cost = 0;"]
        full_exploration_commands = ["SET geqo = off;", "SET join_collapse_limit = 1000;"]
        full_exploration_reduce_random_page_cost_commands = ["SET random_page_cost = 1.1;", "SET geqo = off;", "SET join_collapse_limit = 1000;"]
        full_exploration_no_page_cost_commands = ["SET random_page_cost = 0;", "SET seq_page_cost = 0;", "SET geqo = off;", "SET join_collapse_limit = 1000;"]
        self._engines: List[PostgreSQLExecutionEngine] = [#PostgreSQLExecutionEngine(schema, 0, timeout=timeout),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=operator_disable_commands),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=reduce_random_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=no_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=full_exploration_commands),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=full_exploration_reduce_random_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 0, timeout=timeout, additional_set_commands=full_exploration_no_page_cost_commands),
                                                          PostgreSQLExecutionEngine(schema, 2, timeout=timeout),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=operator_disable_commands),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=reduce_random_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=no_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=full_exploration_commands),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=full_exploration_reduce_random_page_cost_commands),
                                                          #PostgreSQLExecutionEngine(schema, 2, timeout=timeout, additional_set_commands=full_exploration_no_page_cost_commands)
                                                         ]

    def run(self):
        for i, postgresql_execution_engine in enumerate(self._engines):
            self._logger.info("Running PostgreSQL baselines %d" % i)
            set_commands = " ".join(postgresql_execution_engine.set_commands)
            self._logger.info("Additional set commands: %s" % set_commands)

            end_to_end_data_list = []
            for j, test_set in enumerate(self.test_sets):
                self._logger.info("Running test set %d with %d queries" % (j, len(test_set)))
                postgresql_path = self.result_path() + "/postgresql_%d_%d.csv" % (i, j)
                postgresql_file = open(postgresql_path, "w")
                dummy_optimizer = DummyOptimizer()
                rand = np.random.RandomState(42)
                for k in range(self.repetitions):
                    shuffled_test_set = test_set.copy()
                    rand.shuffle(shuffled_test_set)
                    for benchmark_query in shuffled_test_set:
                        end_to_end_data = self._optimize_execute_memorize(dummy_optimizer, [], postgresql_execution_engine, benchmark_query, postgresql_file)
                        end_to_end_data_list.append(end_to_end_data)
                self._logger.info("")
                self._logger.info(self._print_end_to_end_stats(end_to_end_data_list))
                self._logger.info("")








