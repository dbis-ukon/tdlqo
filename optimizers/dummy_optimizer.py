import logging
from typing import Optional

from optimizers.optimizer import Optimizer
from queries.benchmark_query import BenchmarkQuery
from queries.query import Query
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class DummyOptimizer(Optimizer):
    def __init__(self):
        super().__init__(type_name="Dummy Optimizer", name="Dummy Optimizer", description="A dummy optimizer that does nothing.")

    def optimize(self, query: Query, explore: bool = False, benchmark_query: Optional[BenchmarkQuery] = None, debug_logger: Optional[logging.Logger] = None) -> Optional[RelationalAlgebraExpression]:
        return None

