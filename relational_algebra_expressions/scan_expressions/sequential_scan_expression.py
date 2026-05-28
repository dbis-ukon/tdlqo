from typing import Optional

from queries.query import Query
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression


class SequentialScanExpression(ScanExpression):
    def __init__(self, table_occurrence, query: Optional[Query] = None):
        super().__init__(table_occurrence, query=query)

    def string(self) -> str:
        return "SeqScan({})".format(self.table_occurrence.table().name())

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        return isinstance(other, SequentialScanExpression) and self.query() == other.query()



