import time

from optimizers.top_down_random_optimizer import TopDownRandomOptimizer
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.benchmark_schemas import imdb_schema


def enumerate_subplan_queries_test():
    schema = imdb_schema(port=5443)
    query_db = QueryDB("query_db", 5443)
    benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)

    top_down_optimizer = TopDownRandomOptimizer([HashJoinExpression, SequentialScanExpression])

    durations = []

    for benchmark_query in benchmark_queries:
        start_time = time.time()
        subplan_queries = top_down_optimizer.enumerate_logical_groups(benchmark_query.query)
        end_time = time.time()
        duration = end_time - start_time
        print("Enumerated %d subplan queries for query %s in %.2f seconds" % (len(subplan_queries), benchmark_query.query_name(), duration))
        durations.append(duration)

    average_duration = sum(durations) / len(durations) if durations else 0
    print("Average enumeration duration: %.2f seconds" % average_duration)



enumerate_subplan_queries_test()

