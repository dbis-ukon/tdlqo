from typing import Optional, List, Dict

from queries.predicates.conjunction import Conjunction
from queries.predicates.join import Join
from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class JoinExpression(RelationalAlgebraExpression):
    def __init__(self, outer: RelationalAlgebraExpression, inner: RelationalAlgebraExpression, join_condition: Join, query: Optional[Query] = None):
        super().__init__([outer, inner], query=query)
        self.outer: RelationalAlgebraExpression = outer
        self.inner: RelationalAlgebraExpression = inner
        self.join_condition: Join = join_condition

    @staticmethod
    def build_expression(left: RelationalAlgebraExpression, right: RelationalAlgebraExpression, join_condition: Join, query: Optional[Query] = None) -> RelationalAlgebraExpression:
        return JoinExpression(left, right, join_condition, query=query)

    def arity(self) -> int:
        return 2

    def _get_query(self) -> Optional[Query]:
        outer_query = self.outer.query()
        inner_query = self.inner.query()
        if outer_query is None or inner_query is None or not isinstance(outer_query, SPJQuery) or not isinstance(inner_query, SPJQuery):
            return None
        table_occurrences = list(outer_query.table_occurrences()) + list(inner_query.table_occurrences())
        joins = list(outer_query.joins()) + list(inner_query.joins())
        non_equi_join_predicates = list(outer_query.non_equi_join_predicates()) + list(inner_query.non_equi_join_predicates())
        normal_form_join_predicate = self.join_condition.get_normal_form()
        if isinstance(normal_form_join_predicate, Conjunction):
            predicates = normal_form_join_predicate.predicates()
        else:
            predicates = [normal_form_join_predicate]
        for predicate in predicates:
            if isinstance(predicate, Join):
                joins = Join.merge_joins(joins + [predicate])
            else:
                non_equi_join_predicates.append(predicate)
        return SPJQuery(table_occurrences, joins, non_equi_join_predicates)

    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        assert len(new_children) == 2
        new_outer = new_children[0]
        new_inner = new_children[1]
        if new_outer is None and new_inner is None:
            return self
        if new_outer is None:
            new_outer = self.outer
        if new_inner is None:
            new_inner = self.inner
        if equivalent:
            query = self._query
        else:
            query = None
        return JoinExpression(new_outer, new_inner, self.join_condition, query=query)

    def string(self) -> str:
        return f"Join({self.outer.string()}, {self.inner.string()})"
