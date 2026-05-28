from typing import List, Tuple, Dict, Optional, Iterator, Type

from cost_models.local_cost_model import LocalCostModel
from optimizers.top_down_optimizer import TopDownOptimizer
from optimizers.top_down_random_optimizer import TopDownRandomOptimizer
from queries.query import Query
from queries.spj_query import SPJQuery
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements


class _IdentityGroup:
    """Wrapper around GroupRelationalAlgebraExpression that uses table occurrence
    identity (not semantic equality) for hashing and comparison. This prevents
    conflating groups like {t1} and {t2} that reference the same base table
    with the same predicate but are distinct table occurrences in the query."""

    def __init__(self, group: GroupRelationalAlgebraExpression):
        self.group = group
        query = group.query()
        assert isinstance(query, SPJQuery)
        self._table_occurrences = query.table_occurrences()

    def __hash__(self):
        return hash(self._table_occurrences) ^ hash(self.group.requirements)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _IdentityGroup):
            return False
        possibly_equal = self.group.requirements.possibly_equal(other.group.requirements)
        if possibly_equal is False:
            return False
        if self._table_occurrences != other._table_occurrences:
            return False
        if possibly_equal is True:
            return True
        assignment = {to: to for to in self._table_occurrences}
        return self.group.requirements.assignment_equal(other.group.requirements, assignment)


class TopDownCostBasedOptimizer(TopDownOptimizer):
    def __init__(self, cost_model: LocalCostModel, available_operators: List[Type[RelationalAlgebraExpression]]):
        super().__init__("Top-Down Cost-Based Optimizer", "Top-Down Cost-Based Optimizer", "A top-down cost-based optimizer using dynamic programming.", available_operators)
        self._cost_model = cost_model

        self._optimization_memory: Dict[_IdentityGroup, Tuple[RelationalAlgebraExpression, Optional[float]]] = {}

    def _init_optimization(self, query: Query):
        self._optimization_memory = {}

    def _choose(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> RelationalAlgebraExpression:
        key = _IdentityGroup(group)
        if key in self._optimization_memory:
            return self._optimization_memory[key][0]

        chosen_expression, chosen_cost = self._choose_cost(group, memoized_expressions)
        return chosen_expression

    def _choose_cost(self, group: GroupRelationalAlgebraExpression, memoized_expressions: List[RelationalAlgebraExpression]) -> Tuple[RelationalAlgebraExpression, float]:
        lowest_cost = None
        lowest_cost_expression = None
        for memoized_expression in memoized_expressions:
            expression_cost = self._cost_model.local_cost(memoized_expression)
            if expression_cost is None:
                continue
            for child_group in memoized_expression.children:
                assert isinstance(child_group, GroupRelationalAlgebraExpression)
                child_key = _IdentityGroup(child_group)
                if child_key in self._optimization_memory:
                    child_cost = self._optimization_memory[child_key][1]
                else:
                    child_cost = self._choose_cost(child_group, self.enumerate_memoized_expressions(child_group))[1]
                if child_cost is not None:
                    expression_cost += child_cost
                else:
                    expression_cost = None
            if expression_cost is not None and (lowest_cost is None or expression_cost < lowest_cost):
                lowest_cost = expression_cost
                lowest_cost_expression = memoized_expression
        if lowest_cost_expression is None:
            lowest_cost_expression = memoized_expressions[0]
        self._optimization_memory[_IdentityGroup(group)] = (lowest_cost_expression, lowest_cost)
        return lowest_cost_expression, lowest_cost

    def get_all_costs(self, query: SPJQuery) -> Iterator[Tuple[GroupRelationalAlgebraExpression, List[Tuple[RelationalAlgebraExpression, Optional[float]]]]]:
        memory: Dict[GroupRelationalAlgebraExpression, Optional[float]] = {}
        random_optimizer = TopDownRandomOptimizer(self.available_operators)
        random_plan = random_optimizer.optimize(query)
        groups = self._get_all_groups(random_plan)
        for group in groups:
            for (result_group, local_costs), _ in self._get_group_costs(group, memory):
                total_costs = []
                for memoized_expression, local_cost in local_costs:
                    total_cost = local_cost
                    for child_group in memoized_expression.children:
                        assert isinstance(child_group, GroupRelationalAlgebraExpression)
                        child_cost = memory.get(child_group, None)
                        if total_cost is not None:
                            if child_cost is not None:
                                total_cost += child_cost
                            else:
                                total_cost = None
                    total_costs.append((memoized_expression, total_cost))
                yield result_group, total_costs

    def get_all_local_costs(self, query: SPJQuery) -> Iterator[Tuple[Tuple[GroupRelationalAlgebraExpression, List[Tuple[RelationalAlgebraExpression, Optional[float]]]], Dict[GroupRelationalAlgebraExpression, Optional[float]]]]:
        memory: Dict[GroupRelationalAlgebraExpression, Optional[float]] = {}
        random_optimizer = TopDownRandomOptimizer(self.available_operators)
        random_plan = random_optimizer.optimize(query)
        groups = self._get_all_groups(random_plan)
        for group in groups:
            for result in self._get_group_costs(group, memory):
                yield result

    def _get_group_costs(self, group: GroupRelationalAlgebraExpression, memory: Dict[GroupRelationalAlgebraExpression, Optional[float]]) -> Iterator[Tuple[Tuple[GroupRelationalAlgebraExpression, List[Tuple[RelationalAlgebraExpression, Optional[float]]]], Dict[GroupRelationalAlgebraExpression, Optional[float]]]]:
        if group in memory:
            return
        memoized_expressions = self.enumerate_memoized_expressions(group)
        local_costs = []
        lowest_total_cost = None
        for memoized_expression in memoized_expressions:
            local_cost = self._cost_model.local_cost(memoized_expression)
            total_cost = local_cost
            for child_group in memoized_expression.children:
                assert isinstance(child_group, GroupRelationalAlgebraExpression)
                for result in self._get_group_costs(child_group, memory):
                    yield result
                assert child_group in memory
                child_cost = memory[child_group]
                if total_cost is not None:
                    if child_cost is not None:
                        total_cost += child_cost
                    else:
                        total_cost = None
            local_costs.append((memoized_expression, local_cost))
            if total_cost is not None and (lowest_total_cost is None or total_cost < lowest_total_cost):
                lowest_total_cost = total_cost
        memory[group] = lowest_total_cost
        yield (group, local_costs), memory

    @staticmethod
    def _get_all_groups(plan: RelationalAlgebraExpression) -> List[GroupRelationalAlgebraExpression]:
        groups = []
        for child in plan.children:
            groups.extend(TopDownCostBasedOptimizer._get_all_groups(child))
        query = plan.query()
        if isinstance(query, SPJQuery):
            group = GroupRelationalAlgebraExpression(plan.query(), Requirements())  # TODO: compute requirements
            groups.append(group)
        return groups
