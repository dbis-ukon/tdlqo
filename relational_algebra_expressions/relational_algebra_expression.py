
from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Dict

from queries.query import Query
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence


class RelationalAlgebraExpression:
    def __init__(self, children: List[RelationalAlgebraExpression], query: Optional[Query] = None):
        self.children: List[RelationalAlgebraExpression] = children
        assert self.arity() == len(children)
        self._query: Optional[Query] = query
        self._canonical_query: Optional[Query] = None

    @abstractmethod
    def arity(self) -> int:
        raise NotImplementedError()

    def query(self) -> Query:
        if self._query is None:
            self._query = self._get_query()
        return self._query

    def canonical_query(self) -> Query:
        if self._canonical_query is not None:
            return self._canonical_query
        return self.query()

    @abstractmethod
    def _get_query(self) -> Query:
        raise NotImplementedError()

    @abstractmethod
    def replace_children(self, new_children: List[Optional[RelationalAlgebraExpression]], equivalent: bool) -> RelationalAlgebraExpression:
        raise NotImplementedError()

    @abstractmethod
    def string(self) -> str:
        pass

    def syntactic_equality(self, other: RelationalAlgebraExpression) -> bool:
        raise NotImplementedError()

    def canonicalize(self, canonical_queries: Dict[Query, Query]):
        for child in self.children:
            child.canonicalize(canonical_queries)
        if self._query is None:
            self._query = self._get_query()
        canonical_query = canonical_queries.get(self._query, None)
        assert isinstance(self._query, SPJQuery)
        if canonical_query is not None:
            self._canonical_query = canonical_query
        else:
            canonical_queries[self._query] = self._query

    def subplan_queries(self) -> List[Query]:
        queries = [self.query()]
        for child in self.children:
            queries.extend(child.subplan_queries())
        return queries
