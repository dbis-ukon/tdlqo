from typing import List, Tuple, Optional, Dict

from queries.predicates.join import Join
from queries.query import Query
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class SortExpression(RelationalAlgebraExpression):
    def __init__(self, child: RelationalAlgebraExpression, sort_key: List[Tuple[Join, bool]], query: Optional[Query] = None):
        super().__init__([child], query=query)
        self.child: RelationalAlgebraExpression = child
        self.sort_key: List[Tuple[Join, bool]] = sort_key

    def arity(self) -> int:
        return 1

    def _get_query(self) -> Optional[Query]:
        return self.child.query()

    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        assert len(new_children) == 1
        new_child = new_children[0]
        if new_child is None:
            return self
        if equivalent:
            query = self._query
        else:
            query = None
        return SortExpression(new_child, self.sort_key, query=query)

    def string(self) -> str:
        return f"Sort({self.child.string()})"

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        if not isinstance(other, SortExpression) or self.child.syntactic_equality(other.child) or len(self.sort_key) != len(other.sort_key):
            return False
        for own_key, other_key in zip(self.sort_key, other.sort_key):
            if own_key != other_key:
                return False
        return True

