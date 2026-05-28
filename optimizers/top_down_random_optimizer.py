import random
from typing import List, Type

from optimizers.top_down_optimizer import TopDownOptimizer
from queries.query import Query
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class TopDownRandomOptimizer(TopDownOptimizer):
    def __init__(self, available_operators: List[Type[RelationalAlgebraExpression]]):
        super().__init__("Top-Down Random Optimizer", "Top-Down Random Optimizer", "Randomly selects a plan from the memoized expressions.", available_operators)

    def _choose(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> RelationalAlgebraExpression:
        sorted_memoized_expressions = sorted(memoized_expressions, key=lambda e: e.string())
        return random.choice(sorted_memoized_expressions)

