from typing import List

from execution_engines.execution_engine import ExecutionEngine
from experiments.experiment import Experiment
from optimizers.optimizer import Optimizer
from queries.query import Query


class OnlineExperiment(Experiment):
    def __init__(self,
                 optimizer: Optimizer,
                 execution_engine: ExecutionEngine,
                 test_sets: List[List[Query]]):
        super().__init__("Simple Experiment", "")
        self.optimizer = optimizer
        self.execution_engine = execution_engine
        self.test_sets = test_sets

    def run(self):
        for test_queries in self.test_sets:
            for query in test_queries:
                expression = self.optimizer.optimize(query)
                execution_data = self.execution_engine.execute(query, expression)
                if execution_data is not None:
                    print(f"Execution time: {execution_data.execution_time}, Planning time: {execution_data.planning_time}, Timeout: {execution_data.timeout}")



