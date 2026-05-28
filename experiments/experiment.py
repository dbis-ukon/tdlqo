import datetime
from abc import abstractmethod
import logging
from typing import List, Optional, TextIO, Tuple
import numpy as np
import os

from execution_engines.end_to_end_data import EndToEndData
from execution_engines.execution_data import ExecutionData
from execution_engines.execution_engine import ExecutionEngine
from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery


class Experiment:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = datetime.datetime.now()
        os.makedirs(self.result_path(), exist_ok=False)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.result_path() + "/experiment.log")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        handler_print = logging.StreamHandler()
        handler_print.setLevel(logging.INFO)
        logger.addHandler(handler_print)
        logger.info(f"Experiment '{self.name}' created at {self.start_time}")
        self._logger = logger

        self._debug_logger: Optional[logging.Logger] = None

        self._execution_error_logger = logging.getLogger("Execution Errors")
        self._execution_error_logger.setLevel(logging.INFO)
        execution_error_handler = logging.FileHandler(self.result_path() + "/execution_errors.log")
        execution_error_handler.setLevel(logging.INFO)
        self._execution_error_logger.addHandler(execution_error_handler)
        handler_print = logging.StreamHandler()
        handler_print.setLevel(logging.INFO)
        self._execution_error_logger.addHandler(handler_print)

    def result_path(self) -> str:
        return f"results/{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}"

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def _print_end_to_end_stats(end_to_end_data_list: List[EndToEndData]) -> str:
        optimization_times = []
        memorization_times = []
        for end_to_end_data in end_to_end_data_list:
            if end_to_end_data.optimization_time is not None:
                optimization_times.append(end_to_end_data.optimization_time)
            if end_to_end_data.memorization_time is not None:
                memorization_times.append(end_to_end_data.memorization_time)

        result_string = Experiment._print_execution_stats([end_to_end_data.execution_data for end_to_end_data in end_to_end_data_list])
        if len(optimization_times) > 0:
            result_string += "Mean optimization time: %.1f ms\n" % (sum(optimization_times) / len(optimization_times))
            result_string += "Geometric mean optimization time: %.1f ms\n" % (np.exp(np.mean(np.log(optimization_times))))
        if len(memorization_times) > 0:
            result_string += "Mean memorization time: %.1f ms\n" % (sum(memorization_times) / len(memorization_times))
            result_string += "Geometric mean memorization time: %.1f ms\n" % (np.exp(np.mean(np.log(memorization_times))))
        return result_string

    @staticmethod
    def _print_execution_stats(execution_data_list: List[ExecutionData]) -> str:
        effective_execution_times = []
        planning_times = []
        timeouts = 0
        for execution_data in execution_data_list:
            if execution_data.execution_time is not None:
                effective_execution_times.append(execution_data.execution_time)
            else:
                assert execution_data.timeout is not None
                effective_execution_times.append(execution_data.timeout)
                timeouts += 1
            if execution_data.planning_time is not None:
                planning_times.append(execution_data.planning_time)

        result_string = ""
        result_string += "Timeouts: %d/%d\n" % (timeouts, len(execution_data_list))
        if len(effective_execution_times) > 0:
            result_string += "Mean execution time: %.1f ms\n" % (sum(effective_execution_times) / len(effective_execution_times))
            result_string += "Geometric mean execution time: %.1f ms\n" % (np.exp(np.mean(np.log(effective_execution_times))))
        if len(planning_times) > 0:
            result_string += "Mean PostgreSQL planning time: %.1f ms\n" % (sum(planning_times) / len(planning_times))
            result_string += "Geometric mean PostgreSQL planning time: %.1f ms\n" % (np.exp(np.mean(np.log(planning_times))))
        return result_string

    def _optimize_execute_memorize(self,
                                   main_optimizer: Optimizer,
                                   memory_optimizers: List[Optimizer],
                                   execution_engine: ExecutionEngine,
                                   benchmark_query: BenchmarkQuery,
                                   training_iteration_file: TextIO,
                                   explore: bool = False) -> EndToEndData:
        query = benchmark_query.query
        if self._debug_logger is not None:
            self._debug_logger.info("=== optimize %s (query_id=%d, explore=%s) ===" % (benchmark_query.query_name(), benchmark_query.query_id, explore))
        optimization_start_time = datetime.datetime.now()
        expression = main_optimizer.optimize(query, explore=explore, benchmark_query=benchmark_query, debug_logger=self._debug_logger)
        optimization_end_time = datetime.datetime.now()
        if self._debug_logger is not None:
            execution_query = execution_engine.execution_query(benchmark_query.query_text, expression, "EXPLAIN (ANALYZE TRUE, VERBOSE TRUE, FORMAT JSON)")
            self._debug_logger.info("Execution query:\n%s" % execution_query)
        optimization_time = (optimization_end_time - optimization_start_time).total_seconds() * 1000
        execution_data = execution_engine.execute(query, expression, benchmark_query=benchmark_query)
        memorization_time = 0
        for memory_optimizer in memory_optimizers:
            if memory_optimizer == main_optimizer:
                memorization_start_time = datetime.datetime.now()
                main_optimizer.add_to_training_data(execution_data)
                memorization_end_time = datetime.datetime.now()
                memorization_time = (memorization_end_time - memorization_start_time).total_seconds() * 1000
            else:
                memory_optimizer.add_to_training_data(execution_data)
        end_to_end_data = EndToEndData(main_optimizer.optimizer_id, execution_data, optimization_time, memorization_time)
        self._logger.info("%s: %s" % (benchmark_query.query_name(), end_to_end_data.to_string()))
        training_iteration_file.write("%s,%d,%s\n" % (benchmark_query.query_name(), benchmark_query.query_id, end_to_end_data.to_csv()))
        return end_to_end_data

    @staticmethod
    def _cross_validation_splits(benchmark_queries: List[BenchmarkQuery], k: int, repetitions: int) -> List[Tuple[List[BenchmarkQuery], List[BenchmarkQuery]]]:
        assert len(benchmark_queries) > 0
        assert k > 1
        assert repetitions > 0
        rand = np.random.RandomState(42)
        splits = []
        for _ in range(repetitions):
            shuffled_queries = benchmark_queries.copy()
            rand.shuffle(shuffled_queries)
            for i in range(k):
                training_set = []
                test_set = []
                for j, benchmark_query in enumerate(shuffled_queries):
                    if j % k == i:
                        test_set.append(benchmark_query)
                    else:
                        training_set.append(benchmark_query)
                splits.append((training_set, test_set))
        return splits





