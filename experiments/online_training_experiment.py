from itertools import cycle, islice
from typing import List, Callable
import json
import logging
import time

from execution_engines.execution_engine import ExecutionEngine
from execution_engines.postgresql_execution_engine import PostgreSQLExecutionEngine
from experiments.experiment import Experiment
from optimizers.dummy_optimizer import DummyOptimizer
from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery
from queries.query import Query



class OnlineTrainingExperiment(Experiment):
    def __init__(self,
                 optimizer_generator: Callable[[List[BenchmarkQuery], List[BenchmarkQuery]], Optimizer],
                 execution_engine: ExecutionEngine,
                 iterations: int,
                 training_queries: List[BenchmarkQuery],
                 k: int,
                 repetitions: int,
                 always_test: bool,
                 test_sets: List[List[BenchmarkQuery]],
                 queries_per_iteration: int = 100,
                 debug_mode: bool = False):
        super().__init__("Online Training Experiment", "")
        self.optimizer_generator = optimizer_generator
        self.execution_engine = execution_engine
        self.iterations = iterations
        self.training_queries = training_queries
        self.k = k
        self.repetitions = repetitions
        self.always_test = always_test
        self.test_sets = test_sets
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
        cross_validation_splits = self._cross_validation_splits(self.training_queries, self.k, self.repetitions)
        for i, (training_queries, test_queries) in enumerate(cross_validation_splits):
            optimizer = self.optimizer_generator(training_queries + test_queries, training_queries)
            optimizer_start_time = time.time()
            optimizer.print_parameters(self._logger)
            optimizer.optimizer_id = str(i)
            if len(training_queries) > self.queries_per_iteration:
                iteration_training_query_list = []
                it = cycle(training_queries)
                for _ in range(self.iterations):
                    iteration_training_query_list.append(list(islice(it, self.queries_per_iteration)))
            else:
                iteration_training_query_list = [training_queries] * self.iterations
            for iteration, iteration_training_queries in enumerate(iteration_training_query_list):
                self._iteration_train_and_test(iteration, optimizer, iteration_training_queries, [])

                training_meta_data = optimizer.train(logger=self._logger)

                current_time = time.time()
                training_meta_data["optimizer_training_time_seconds"] = current_time - optimizer_start_time
                training_meta_data_path = self.result_path() + "/training_meta_data_iteration_%d_%d.json" % (i, iteration + 1)
                with open(training_meta_data_path, "w") as f:
                    json.dump(training_meta_data, f, indent=4)
                optimizer_path = self.result_path() + "/optimizer_iteration_%d_%d" % (i, iteration + 1)
                optimizer.save(optimizer_path)
                test_start_time = time.time()
                if self.always_test:
                    self._iteration_test(iteration, optimizer, test_queries)
                test_end_time = time.time()
                optimizer_start_time = optimizer_start_time + (test_end_time - test_start_time)
            self._iteration_train_and_test(self.iterations, optimizer, training_queries, test_queries)

            for j, test_queries in enumerate(self.test_sets):
                test_path = self.result_path() + "/test_set_%d.csv" % j
                test_file = open(test_path, "w")
                self._logger.info("Running test set with %d queries" % len(test_queries))
                execution_data_list = []
                for benchmark_query in test_queries:
                    query = benchmark_query.query
                    if self._debug_logger is not None:
                        self._debug_logger.info("=== optimize %s (query_id=%d, test_set=%d) ===" % (benchmark_query.query_name(), benchmark_query.query_id, j))
                    expression = optimizer.optimize(query, benchmark_query=benchmark_query, debug_logger=self._debug_logger)
                    if self._debug_logger is not None:
                        execution_query = self.execution_engine.execution_query(benchmark_query.query_text, expression, "EXPLAIN (ANALYZE TRUE, VERBOSE TRUE, FORMAT JSON)")
                        self._debug_logger.info("Execution query:\n%s" % execution_query)
                    execution_data = self.execution_engine.execute(query, expression)
                    execution_data_list.append(execution_data)
                    self._logger.info("%s: %s" % (benchmark_query.query_name(), execution_data.to_string()))
                    test_file.write("%s,%d,%s\n" % (benchmark_query.query_name(), benchmark_query.query_id, execution_data.to_csv()))
                test_file.close()
                self._logger.info("")
                self._logger.info(self._print_execution_stats(execution_data_list))
                self._logger.info("")

    def _iteration_train_and_test(self, iteration: int, optimizer: Optimizer, training_queries: List[BenchmarkQuery], test_queries: List[BenchmarkQuery]):
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

        self._iteration_test(iteration, optimizer, test_queries)

    def _iteration_test(self, iteration: int, optimizer: Optimizer, test_queries: List[BenchmarkQuery]):
        if len(test_queries) == 0:
            return
        test_end_to_end_data_list = []

        test_iteration_path = self.result_path() + "/test_iteration_%d.csv" % iteration
        test_iteration_file = open(test_iteration_path, "a")

        for benchmark_query in test_queries:
            end_to_end_data = self._optimize_execute_memorize(optimizer, [], self.execution_engine, benchmark_query, test_iteration_file, explore=False)
            test_end_to_end_data_list.append(end_to_end_data)
        test_iteration_file.close()

        self._logger.info("")
        self._logger.info("Test iteration %d completed" % iteration)
        self._logger.info(self._print_end_to_end_stats(test_end_to_end_data_list))
        self._logger.info("")















