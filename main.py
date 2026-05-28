
import os

from experiments.postgresql_cardinality_experiment import PostgreSQLCardinalityExperiment

os.environ["WANDB_MODE"] = "disabled"

from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from experiments.consistency_experiment import ConsistencyExperiment
from experiments.online_training_experiment import OnlineTrainingExperiment
from experiments.postgresql_experiment import PostgreSQLExperiment
from experiments.true_cardinality_experiment import TrueCardinalityExperiment
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import TopDownLearnedOptimizerConfiguration
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from queries.spj_query import SPJQuery
from schemas.benchmark_schemas import imdb_schema, stack_schema

query_db = QueryDB("query_db", 5443)
schema = stack_schema(port=5443)
benchmark_queries = load_benchmark(query_db, 22)
parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
benchmark_queries = [benchmark_query for benchmark_query in benchmark_queries if benchmark_query.query is not None]
timeout = 2 * 60 * 1000
execution_engine = PgHintPlanExecutionEngine(schema, 2, timeout=timeout)

tdlqo_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.jgmp_sized_neural_network_configuration(schema))

tdlqo_ncd_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.jgmp_sized_neural_network_no_cardinality_deduction_configuration(schema))
tdlqo_ndc_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.jgmp_sized_neural_network_no_decomposed_costs_configuration(schema))


online_training_experiment = OnlineTrainingExperiment(tdlqo_constructor, execution_engine, 20, benchmark_queries, 5, 3, True, [])
online_training_experiment.run()
