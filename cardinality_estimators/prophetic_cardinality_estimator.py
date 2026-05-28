from typing import Dict, Optional, Tuple

from cardinality_estimators.cardinality_estimator import CardinalityEstimator, CardinalityMode
from cardinality_estimators.cardinality_range import CardinalityRange
from queries.predicates.true_predicate import TruePredicate
from queries.query import Query
from queries.spj_query import SPJQuery
from schemas.schema import Schema


class PropheticCardinalityEstimator(CardinalityEstimator):
    def __init__(self, gather_on_demand_schema: Optional[Schema] = None, raise_inconsistent_merges: bool = False):
        self.memory: Dict[Query, CardinalityRange] = {}
        self.gather_on_demand_schema = gather_on_demand_schema
        self.raise_inconsistent_merges = raise_inconsistent_merges

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
                cursor = self.gather_on_demand_schema.connection().cursor()
                cardinality_query = query.get_query_text("COUNT(*)")
                cursor.execute(cardinality_query)
                result = cursor.fetchone()
                cursor.close()
                cardinality = result[0]
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




