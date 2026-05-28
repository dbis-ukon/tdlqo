from typing import List, Optional

from queries.query import Query
from queries.spj_query import SPJQuery
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.requirements import Requirements


class GroupRelationalAlgebraExpression(RelationalAlgebraExpression):
    def __init__(self, query: Query, requirements: Requirements):
        super().__init__([], query=query)
        self.requirements = requirements

    def arity(self) -> int:
        return 0

    def string(self) -> str:
        query = self.query()
        assert isinstance(query, SPJQuery)
        tables = sorted(
            (to.table().name(), to.alias() or "") for to in query.table_occurrences()
        )
        parts = [f"{name}:{alias}" for name, alias in tables]
        req = ""
        if self.requirements.force_index_scan is not None:
            index, memoized = self.requirements.force_index_scan
            req = f",idx={index.name()},memo={memoized}"
        return "Group(" + ",".join(parts) + req + ")"

    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        return self

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        return self == other

    def __hash__(self):
        query_hash = hash(self.query())
        requirements_hash = hash(self.requirements)
        return query_hash ^ requirements_hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, GroupRelationalAlgebraExpression):
            return False
        possibly_equal_requirements = self.requirements.possibly_equal(other.requirements)
        if possibly_equal_requirements is False:
            return False
        own_query = self.query()
        other_query = other.query()
        if not (isinstance(own_query, SPJQuery) and isinstance(other_query, SPJQuery)):
            return False

        assignment = own_query.known_equal_assignment(other_query)
        if assignment is None:
            return False
        if possibly_equal_requirements is True:
            return True
        return self.requirements.assignment_equal(other.requirements, assignment)





