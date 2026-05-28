from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from schemas.benchmark_schemas import imdb_schema


def save_and_load_test():
    schema = imdb_schema(port=5443)
    query_db = QueryDB("query_db", 5443)
    benchmark_queries = load_benchmark(query_db, 2)[:20]
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    execution_engine = PgHintPlanExecutionEngine(schema, 2, timeout=2 * 60 * 1000)

    optimizer = TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.default_configuration(schema))
    optimizer.explore_and_train([benchmark_query.query for benchmark_query in benchmark_queries], 10 * 60)
    optimizer.save("/tmp/test_optimizer")
    loaded_optimizer = TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.default_configuration(schema))
    loaded_optimizer.load("/tmp/test_optimizer")
    execution_times = []
    for benchmark_query in benchmark_queries:
        plan = optimizer.optimize(benchmark_query.query)
        execution_data = execution_engine.execute(benchmark_query.query, plan, analyze=True)
        execution_times.append(execution_data.execution_time)
    print("Execution times before loading: ", execution_times)
    loaded_execution_times = []
    for benchmark_query in benchmark_queries:
        plan = loaded_optimizer.optimize(benchmark_query.query)
        execution_data = execution_engine.execute(benchmark_query.query, plan, analyze=True)
        loaded_execution_times.append(execution_data.execution_time)
    print("Execution times after loading: ", loaded_execution_times)




save_and_load_test()











