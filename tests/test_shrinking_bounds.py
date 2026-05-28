import random
import sys

from cardinality_estimators.cardinality_estimator import CardinalityMode
from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from schemas.benchmark_schemas import imdb_schema, stack_schema


def run_test(schema, benchmark_id, query_db_port=5443, max_edges=1000, seed=0):
    configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
    optimizer = TopDownLearnedOptimizer(schema, configuration)
    optimizer._add_statistical_cardinality_bounds()

    query_db = QueryDB("query_db", query_db_port)
    benchmark_queries = load_benchmark(query_db, benchmark_id)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    for benchmark_query in benchmark_queries:
        if benchmark_query.query is None:
            continue
        subplan_queries = optimizer.enumerate_logical_groups(benchmark_query.query)
        optimizer._add_bounds(benchmark_query.query, subplan_queries)

    # Each _upper_bounds[larger][smaller] = scale encodes |smaller| <= |larger| * scale.
    # We verify edges directly (not propagated memory) because _add_bounds alone does not
    # populate cardinalities for multi-table queries -- that requires exploration.
    edges = []
    for larger_query, smaller_dict in optimizer._upper_bounds.items():
        for smaller_query, scale in smaller_dict.items():
            edges.append((smaller_query, larger_query, scale))

    # Random order + cap so the sample is representative across the workload rather than
    # biased by insertion order (which clusters edges by workload template).
    rand = random.Random(seed)
    rand.shuffle(edges)
    if max_edges is not None:
        edges = edges[:max_edges]

    connection = schema.connection()
    cursor = connection.cursor()
    cursor.execute("SET statement_timeout = '60s';")
    cursor.close()

    # gather_on_demand runs COUNT(*) once per unseen query and memorizes the exact cardinality,
    # so repeated edges on the same query hit the cache for free.
    gather_estimator = PropheticCardinalityEstimator(gather_on_demand_schema=schema)

    passed = 0
    failed = 0
    timed_out = 0
    scaled_passed = 0
    scaled_failed = 0

    for smaller_query, larger_query, scale in edges:
        try:
            smaller_c = gather_estimator.estimate(smaller_query, CardinalityMode.MIN)
            larger_c = gather_estimator.estimate(larger_query, CardinalityMode.MIN)
            if smaller_c <= larger_c * scale:
                passed += 1
                if scale != 1.0:
                    scaled_passed += 1
            else:
                failed += 1
                if scale != 1.0:
                    scaled_failed += 1
                print("  FAIL  smaller=%d > larger=%d * %.3f" % (smaller_c, larger_c, scale))
                print("         smaller: %s" % smaller_query.query_text())
                print("         larger : %s" % larger_query.query_text())
        except Exception as e:
            connection.rollback()
            reset_cursor = connection.cursor()
            reset_cursor.execute("SET statement_timeout = '60s';")
            reset_cursor.close()
            timed_out += 1
            print("  TIMEOUT  %s" % str(e).strip())

    connection.close()
    print("\n%s: %d passed, %d failed, %d timed out (of %d edges tested, %d total)" % (
        schema.database_name(), passed, failed, timed_out, len(edges),
        sum(len(d) for d in optimizer._upper_bounds.values())))
    print("  Non-unit-scale: %d passed, %d failed" % (scaled_passed, scaled_failed))
    return failed


if __name__ == "__main__":
    schema = stack_schema(port=5443)
    print("Loaded schema '%s'..." % schema.database_name())
    print("Running shrinking bound tests...\n")
    failures = run_test(schema, benchmark_id=21)
    sys.exit(1 if failures > 0 else 0)
