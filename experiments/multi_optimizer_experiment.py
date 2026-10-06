from typing import List

from execution_engines.execution_engine import ExecutionEngine
from experiments.experiment import Experiment
from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery


class MultiOptimizerExperiment(Experiment):
    """Evaluates multiple optimizers, each only on its own test set."""

    def __init__(self,
                 optimizers: List[Optimizer],
                 execution_engine: ExecutionEngine,
                 test_sets: List[List[BenchmarkQuery]],
                 repeated_executions: int = 1):
        super().__init__("Multi Optimizer Experiment", "")
        assert len(optimizers) == len(test_sets)
        self.optimizers = optimizers
        self.execution_engine = execution_engine
        self.test_sets = test_sets
        self.repeated_executions = repeated_executions

    def run(self):
        for i, (optimizer, test_queries) in enumerate(zip(self.optimizers, self.test_sets)):
            optimizer.optimizer_id = str(i)
            test_path = self.result_path() + "/test_set_%d.csv" % i
            test_file = open(test_path, "w")
            self._logger.info("Running test set %d with %d queries" % (i, len(test_queries)))
            end_to_end_data_list = []
            for _ in range(self.repeated_executions):
                for benchmark_query in test_queries:
                    end_to_end_data = self._optimize_execute_memorize(optimizer, [], self.execution_engine, benchmark_query, test_file)
                    end_to_end_data_list.append(end_to_end_data)
            test_file.close()
            self._logger.info("")
            self._logger.info(self._print_end_to_end_stats(end_to_end_data_list))
            self._logger.info("")
