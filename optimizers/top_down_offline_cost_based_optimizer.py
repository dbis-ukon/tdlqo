import datetime
import logging
from typing import List, Type, Optional

from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.local_cost_model import LocalCostModel
from execution_engines.execution_data import ExecutionData
from execution_engines.postgresql_execution_engine import PostgreSQLExecutionEngine
from optimizers.top_down_cost_based_optimizer import TopDownCostBasedOptimizer
from queries.query import Query
from queries.spj_query import SPJQuery
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from schemas.schema import Schema


class TopDownOfflineCostBasedOptimizer(TopDownCostBasedOptimizer):
    def __init__(self, schema: Schema, cost_model: LocalCostModel, available_operators: List[Type[RelationalAlgebraExpression]]):
        super().__init__(cost_model, available_operators)
        self._schema: Schema = schema

    def add_to_training_data(self, execution_data: ExecutionData):
        cardinality_estimator = self._cost_model.cardinality_estimator
        assert isinstance(cardinality_estimator, PropheticCardinalityEstimator)
        for query, (cardinality_range, _) in execution_data.cardinality_ranges.items():
            cardinality_estimator.memorize(query, cardinality_range)

    def explore(self, queries: List[Query], logger: Optional[logging.Logger] = None):
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        logger.info("Exploring %d queries - %s" % (len(queries), datetime.datetime.utcnow().isoformat()))
        spj_queries = [query for query in queries if isinstance(query, SPJQuery)]
        logger.info("Enumerating groups for %d queries." % len(spj_queries))
        all_query_groups = []
        all_groups = set()
        for query in spj_queries:
            query_groups = self.enumerate_logical_groups(query)
            for group in query_groups:
                all_groups.add(group)
            all_query_groups.append(query_groups)
        logger.info("Enumerated %d distinct groups." % len(all_groups))
        for query_groups in all_query_groups:
            query_groups.sort(key=lambda q: -len(q.table_occurrences()))

        execution_engine = PostgreSQLExecutionEngine(self._schema, 0, timeout=60 * 1000)
        cardinality_estimator = self._cost_model.cardinality_estimator
        assert isinstance(cardinality_estimator, PropheticCardinalityEstimator)
        execution_count = 0
        while len(all_query_groups) > 0:
            query_groups = all_query_groups.pop(0)
            query_group = query_groups.pop(0)
            if len(query_groups) > 0:
                all_query_groups.append(query_groups)
            if query_group in cardinality_estimator.memory:
                cardinality_range = cardinality_estimator.memory[query_group]
                if cardinality_range.is_exact():
                    continue
            execution_data = execution_engine.execute(query_group, None)
            execution_count += 1
            self.add_to_training_data(execution_data)
            if execution_data.timeout is not None:
                logger.info("Execution %d timed out." % execution_count)
            else:
                logger.info("Execution %d completed - %d cardinalities in memory." % (
                execution_count, len(cardinality_estimator.memory)))

        logger.info("Exploration completed - %s" % datetime.datetime.utcnow().isoformat())







