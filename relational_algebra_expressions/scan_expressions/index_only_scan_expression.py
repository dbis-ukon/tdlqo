from typing import Optional

from queries.query import Query
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from schemas.index import Index


class IndexOnlyScanExpression(ScanExpression):
    def __init__(self, table_occurrence: TableOccurrence, index: Optional[Index], query: Optional[Query] = None):
        super().__init__(table_occurrence, query=query)
        self.index: Index = index
        if index:
            assert index in table_occurrence.table().indexes()

    def string(self) -> str:
        return "IndexOnlyScan({})".format(self.table_occurrence.table().name())



