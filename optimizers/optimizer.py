from abc import abstractmethod
from typing import Optional, List, Tuple
import logging
import time
import json

from execution_engines.execution_data import ExecutionData
from queries.benchmark_query import BenchmarkQuery
from queries.query import Query
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class Optimizer:
    def __init__(self, type_name: str, name: str, description: str, optimizer_id: Optional[str] = None):
        self.type_name: str = type_name
        self.name: str = name
        self.description: str = description
        self.optimizer_id: Optional[str] = optimizer_id
        # SET commands the execution engine applies when optimize() returns None and the
        # default PostgreSQL plan is executed instead.
        self.default_plan_set_commands: List[str] = []

    @abstractmethod
    def optimize(self, query: Query, explore: bool = False, benchmark_query: Optional[BenchmarkQuery] = None, debug_logger: Optional[logging.Logger] = None) -> Optional[RelationalAlgebraExpression]:
        pass

    def add_to_training_data(self, execution_data: ExecutionData):
        pass

    def explore(self, queries: List[Query], time_budget: Optional[float], logger: Optional[logging.Logger] = None) -> bool:
        pass

    def explore_and_train(self, queries: List[Query], time_budget: Optional[float], logger: Optional[logging.Logger] = None, start_time: Optional[float] = None, path: Optional[str] = None) -> Tuple[bool, dict]:
        fully_explored = self.explore(queries, time_budget, logger=logger)
        training_meta_data = self.train(logger=logger)
        if start_time is not None:
            current_time = time.time()
            training_meta_data["optimizer_training_time_seconds"] = current_time - start_time

        if path is not None:
            training_meta_data_path = path + "_meta_data.json"
            with open(training_meta_data_path, "w") as f:
                json.dump(training_meta_data, f, indent=4)
        return fully_explored, training_meta_data

    def train(self, logger: Optional[logging.Logger] = None) -> dict:
        pass

    def print_parameters(self, logger: logging.Logger):
        pass

    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

