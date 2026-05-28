from typing import List

from optimizers.top_down_learned_optimizer.memory.memoized_expression_memory import MemoizedExpressionMemory
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import \
    TopDownLearnedOptimizerData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import PlanType


class GroupMemory:
    def __init__(self, encoding: TopDownLearnedOptimizerData, memoized_expression_memories: List[MemoizedExpressionMemory], subgroup_target_memory_indexes: List[int]):
        # Note that query cardinalities are not stored here, but are stored in the cardinality estimator of the cost model used by optimizer for training.
        self.encoding = encoding
        self.memoized_expression_memories = memoized_expression_memories
        self.memoized_expression_dict = {}
        self.plan_dict = {}
        self.plan_type_counts = {plan_type: 0 for plan_type in PlanType}
        for memoized_expression_memory in memoized_expression_memories:
            self.memoized_expression_dict[memoized_expression_memory.memoized_expression] = memoized_expression_memory
            key = (memoized_expression_memory.plan_type, memoized_expression_memory.plan_index)
            self.plan_dict[key] = memoized_expression_memory
            self.plan_type_counts[memoized_expression_memory.plan_type] += 1
        self.subgroup_target_memory_indexes = subgroup_target_memory_indexes





