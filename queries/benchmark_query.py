
from __future__ import annotations

from typing import Optional
from queries.query import Query


class BenchmarkQuery:
    def __init__(self,
                 query_id: int,
                 query_text: str,
                 query_name: Optional[str] = None,
                 template: Optional[str] = None,
                 query: Optional[Query] = None):
        self.query_id = query_id
        self.query_text = query_text
        self._query_name = query_name
        self.template = template
        self.query = query

    def query_name(self) -> str:
        if self._query_name is None:
            return "Query %d" % self.query_id
        return self._query_name

    def __eq__(self, other: BenchmarkQuery) -> bool:
        return self.query_id == other.query_id

    def __hash__(self) -> int:
        return hash(self.query_id)
