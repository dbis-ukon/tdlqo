from typing import Optional, List

from queries.predicates.join import Join
from queries.query import Query
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class MergeJoinExpression(JoinExpression):
    def __init__(self, outer: RelationalAlgebraExpression, inner: RelationalAlgebraExpression, join_condition: Join, query: Optional[Query] = None):
        super().__init__(outer, inner, join_condition, query=query)

    @staticmethod
    def build_expression(left: RelationalAlgebraExpression, right: RelationalAlgebraExpression, join_condition: Join, query: Optional[Query] = None) -> RelationalAlgebraExpression:
        return MergeJoinExpression(left, right, join_condition, query=query)

    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        assert len(new_children) == 2
        new_left = new_children[0]
        new_right = new_children[1]
        if new_left is None and new_right is None:
            return self
        if new_left is None:
            new_left = self.outer
        if new_right is None:
            new_right = self.inner
        if equivalent:
            query = self._query
        else:
            query = None
        return MergeJoinExpression(new_left, new_right, self.join_condition, query=query)

    def string(self) -> str:
        return f"MergeJoin({self.outer.string()}, {self.inner.string()})"

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        return isinstance(other, MergeJoinExpression) and self.outer.syntactic_equality(other.outer) and self.inner.syntactic_equality(other.inner) and self.query() == other.query()


