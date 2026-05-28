from typing import Optional

from queries.query import Query
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression


class TidScanExpression(ScanExpression):
    def __init__(self, table_occurrence, query: Optional[Query] = None):
        super().__init__(table_occurrence, query=query)

    def string(self) -> str:
        return "TidScan({})".format(self.table_occurrence.table().name())



