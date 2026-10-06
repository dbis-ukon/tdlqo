import json
from typing import Dict, List, Optional, Tuple

import psycopg2

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cardinality_estimators.cardinality_range import CardinalityRange
from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from queries.benchmark_query import BenchmarkQuery
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.spj_query import SPJQuery
from schemas.schema import Schema


class PropheticCardinalityEstimator(CardinalityEstimator):
    def __init__(self, gather_on_demand_schema: Optional[Schema] = None, raise_inconsistent_merges: bool = False, gather_on_demand_timeout_seconds: Optional[float] = None):
        self.memory: Dict[Query, CardinalityRange] = {}
        self.gather_on_demand_schema = gather_on_demand_schema
        self.raise_inconsistent_merges = raise_inconsistent_merges
        self.gather_on_demand_timeout_seconds = gather_on_demand_timeout_seconds
        self._gather_connection = None
        self._gather_connection_timeout_seconds = None
        self._fallback_cardinality_estimator: Optional[PostgreSQLCardinalityEstimator] = None

    def estimate(self, query: Query, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        max_cardinality = None
        if isinstance(query, SPJQuery):
            table_occurrences = query.table_occurrences()
            if len(table_occurrences) == 1:
                table_occurrence = list(table_occurrences)[0]
                if isinstance(table_occurrence.predicate(), TruePredicate):
                    table = table_occurrence.table()
                    return table.cardinality()
                else:
                    max_cardinality = table_occurrence.table().cardinality()
        cardinality_range = self.memory.get(query)
        if cardinality_range is None:
            if self.gather_on_demand_schema is not None and isinstance(query, SPJQuery):
                cardinality = self._gather_cardinality(query)
                if cardinality is None:
                    # The fetch timed out: memorize a PostgreSQL estimate so the count is not attempted again.
                    if self._fallback_cardinality_estimator is None:
                        self._fallback_cardinality_estimator = PostgreSQLCardinalityEstimator(self.gather_on_demand_schema, False)
                    cardinality = self._fallback_cardinality_estimator.estimate(query)
                self.memorize(query, CardinalityRange(cardinality, cardinality))
                return float(cardinality)
            if cardinality_mode == CardinalityMode.MIN:
                return 0.0
            return None
        if cardinality_mode == CardinalityMode.MIN:
            return cardinality_range.min_cardinality
        if cardinality_mode == CardinalityMode.MAX:
            if max_cardinality is not None and (cardinality_range.max_cardinality is None or max_cardinality < cardinality_range.max_cardinality):
                return max_cardinality
            return cardinality_range.max_cardinality
        if cardinality_range.max_cardinality is not None:
            return (self.memory[query].min_cardinality + self.memory[query].max_cardinality) / 2
        return None

    def _gather_cardinality(self, query: SPJQuery) -> Optional[float]:
        """Fetches the true cardinality, returning None if the configured timeout is exceeded."""
        if self._gather_connection is None:
            self._gather_connection = self.gather_on_demand_schema.connection()
            # Autocommit keeps statement_timeout session-scoped: a rollback would revert a transaction-scoped SET.
            self._gather_connection.autocommit = True
        if self._gather_connection_timeout_seconds != self.gather_on_demand_timeout_seconds:
            self._gather_connection_timeout_seconds = self.gather_on_demand_timeout_seconds
            timeout_milliseconds = 0 if self.gather_on_demand_timeout_seconds is None else int(self.gather_on_demand_timeout_seconds * 1000)
            with self._gather_connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = %d;" % timeout_milliseconds)
        try:
            with self._gather_connection.cursor() as cursor:
                cursor.execute(query.get_query_text("COUNT(*)"))
                return cursor.fetchone()[0]
        except psycopg2.errors.QueryCanceled:
            self._gather_connection.rollback()
            return None

    def load_cardinalities(self, path: str, benchmark_queries: List[BenchmarkQuery]):
        """Loads subquery cardinalities exported by export_true_cardinalities.py, keyed by
        benchmark query id and identified by the table occurrence aliases of the parsed queries."""
        with open(path) as f:
            data = json.load(f)
        benchmark_query_dict = {}
        for benchmark_query in benchmark_queries:
            benchmark_query_dict[str(benchmark_query.query_id)] = benchmark_query
        for query_id, entries in data["queries"].items():
            benchmark_query = benchmark_query_dict.get(query_id)
            if benchmark_query is None or benchmark_query.query is None:
                continue
            query = benchmark_query.query
            assert isinstance(query, SPJQuery)
            alias_dict = {table_occurrence.alias(): table_occurrence for table_occurrence in query.table_occurrences()}
            for entry in entries:
                table_occurrences = [alias_dict[alias] for alias in entry["aliases"]]
                subquery = query.induced_subquery(table_occurrences)
                self.memorize(subquery, CardinalityRange(entry["cardinality"], entry["cardinality"]))

    def estimate_range(self, query: Query) -> CardinalityRange:
        return self.memory.get(query, CardinalityRange(0, None))

    def memorize(self, query: Query, cardinality_range: CardinalityRange) -> Tuple[bool, bool]:
        if isinstance(query, SPJQuery) and len(query.table_occurrences()) == 1:
            table_cardinality = list(query.table_occurrences())[0].table().cardinality()
            if cardinality_range.min_cardinality > table_cardinality:
                cardinality_range = CardinalityRange(0, cardinality_range.max_cardinality)
            if cardinality_range.max_cardinality is None or cardinality_range.max_cardinality > table_cardinality:
                cardinality_range = CardinalityRange(cardinality_range.min_cardinality, table_cardinality)
        if query in self.memory:
            old_cardinality_range = self.memory[query]
            try:
                new_cardinality_range = cardinality_range.merge(old_cardinality_range)
            except ValueError:
                error_message = f"Cannot merge cardinality ranges {old_cardinality_range.min_cardinality} - {old_cardinality_range.max_cardinality} with {cardinality_range.min_cardinality} - {cardinality_range.max_cardinality} for query {query.query_text()}."
                if self.raise_inconsistent_merges:
                    raise ValueError(error_message)
                else:
                    # In some rare cases our ExecutionEngine extracts incorrect cardinality ranges
                    # In cases of inconsistencies, we log the error and reset the cardinality range to no information
                    print(error_message)
                    new_cardinality_range = CardinalityRange.no_information_cardinality_range()
            self.memory[query] = new_cardinality_range
            min_cardinality_changed = new_cardinality_range.min_cardinality > old_cardinality_range.min_cardinality
            if new_cardinality_range.max_cardinality is None:
                max_cardinality_changed = False
            elif old_cardinality_range.max_cardinality is None:
                max_cardinality_changed = True
            else:
                max_cardinality_changed = new_cardinality_range.max_cardinality < old_cardinality_range.max_cardinality
            return min_cardinality_changed, max_cardinality_changed
        else:
            self.memory[query] = cardinality_range
            return cardinality_range.min_cardinality > 0, cardinality_range.max_cardinality is not None




