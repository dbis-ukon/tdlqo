from typing import Optional

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from queries.query import Query
from queries.spj_query import SPJQuery
from schemas.schema import Schema


class PostgreSQLCardinalityEstimator(CardinalityEstimator):
    def __init__(self, schema: Schema, cache: bool):
        self._connection = schema.connection()
        self._cache = cache
        self._memory = {}

    def estimate(self, query: Query, cardinality_mode: CardinalityMode = CardinalityMode.MEAN) -> Optional[float]:
        if not isinstance(query, SPJQuery):
            return None
        if self._cache and query in self._memory:
            return self._memory[query]
        cursor = self._connection.cursor()
        sql_query = "EXPLAIN (FORMAT JSON) %s" % query.get_query_text(forced_select_clause="*")
        cursor.execute(sql_query)
        plan = cursor.fetchone()[0]
        cursor.close()
        cardinality_estimate = float(plan[0]['Plan']['Plan Rows'])
        self.memorize(query, cardinality_estimate)
        return cardinality_estimate

    def memorize(self, query: Query, cardinality: float):
        if self._cache:
            self._memory[query] = cardinality





