from typing import Optional, List, Dict

from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class ScanExpression(RelationalAlgebraExpression):
    def __init__(self, table_occurrence: TableOccurrence, query: Optional[Query] = None):
        super().__init__([], query=query)
        self.table_occurrence: TableOccurrence = table_occurrence

    def arity(self) -> int:
        return 0

    def _get_query(self) -> Optional[Query]:
        return SPJQuery([self.table_occurrence], [], [])

    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        return self

    def string(self) -> str:
        return "Scan({})".format(self.table_occurrence.table().name())
