from typing import Optional

from queries.query import Query
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from schemas.index import Index


class IndexScanExpression(ScanExpression):
    def __init__(self, table_occurrence: TableOccurrence, index: Optional[Index], memoized: bool, join_clause: bool, query: Optional[Query] = None):
        super().__init__(table_occurrence, query=query)
        self.index: Optional[Index] = index
        self.memoized: bool = memoized
        self.join_clause: bool = join_clause
        assert index is None or index in table_occurrence.table().indexes()

    def string(self) -> str:
        idx_name = self.index.name() if self.index is not None else "None"
        return "IndexScan({},idx={},memo={},jc={})".format(
            self.table_occurrence.table().name(), idx_name, self.memoized, self.join_clause)

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        return isinstance(other, IndexScanExpression) and self.index == other.index and self.query() == other.query()



