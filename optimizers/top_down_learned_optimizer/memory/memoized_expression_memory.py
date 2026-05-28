from typing import Optional, List

from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import \
    PlanType
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class MemoizedExpressionMemory:
    def __init__(self,
                 memoized_expression: RelationalAlgebraExpression,
                 plan_type: PlanType,
                 plan_index: int,
                 min_local_cost: Optional[float] = None,
                 max_local_cost: Optional[float] = None,
                 target_memory_indexes: Optional[List[int]] = None):
        self.memoized_expression = memoized_expression
        self.plan_type = plan_type
        self.plan_index = plan_index
        self.min_local_cost = min_local_cost
        self.max_local_cost = max_local_cost
        self.target_memory_indexes = target_memory_indexes





