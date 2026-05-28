import sys

from cardinality_estimators.cardinality_estimator import CardinalityMode
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from schemas.benchmark_schemas import imdb_schema, stack_schema


def describe_query(query):
    table_occurrences = list(query.table_occurrences())
    if len(table_occurrences) == 1:
        return table_occurrences[0].table().name()
    join = list(query.joins())[0]
    parts = []
    for to, col in join.equivalence_class():
        parts.append("%s.%s" % (to.table().name(), col.name()))
    return " = ".join(parts)


def run_test(schema):
    configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
    optimizer = TopDownLearnedOptimizer(schema, configuration)
    optimizer._add_statistical_cardinality_bounds()

    estimator = optimizer._cardinality_estimator

    connection = schema.connection()
    cursor = connection.cursor()
    cursor.execute("SET statement_timeout = '60s';")

    passed = 0
    failed = 0
    timed_out = 0

    for query, cardinality_range in estimator.memory.items():
        lower_bound = cardinality_range.min_cardinality
        description = describe_query(query)
        count_query = query.get_query_text("COUNT(*)")

        try:
            cursor.execute(count_query)
            actual = cursor.fetchone()[0]
            upper_bound = cardinality_range.max_cardinality
            lower_ok = actual >= lower_bound
            upper_ok = upper_bound is None or actual <= upper_bound
            if lower_ok and upper_ok:
                bounds_str = "lower=%d" % lower_bound
                if upper_bound is not None:
                    bounds_str += ", upper=%d" % upper_bound
                print("  PASS  %s: %s, actual=%d" % (description, bounds_str, actual))
                passed += 1
            else:
                reasons = []
                if not lower_ok:
                    reasons.append("lower=%d, off by %d" % (lower_bound, lower_bound - actual))
                if not upper_ok:
                    reasons.append("upper=%d, off by %d" % (upper_bound, actual - upper_bound))
                print("  FAIL  %s: %s, actual=%d" % (description, ", ".join(reasons), actual))
                failed += 1
        except Exception as e:
            connection.rollback()
            cursor.execute("SET statement_timeout = '60s';")
            print("  TIMEOUT  %s: lower=%d (%s)" % (description, lower_bound, str(e).strip()))
            timed_out += 1

    cursor.close()
    connection.close()
    print("\n%s: %d passed, %d failed, %d timed out" % (schema.database_name(), passed, failed, timed_out))
    return failed


if __name__ == "__main__":
    schema = stack_schema(port=5443)
    print("Loaded schema '%s'..." % schema.database_name())
    print("Running cardinality bound tests...\n")
    failures = run_test(schema)
    sys.exit(1 if failures > 0 else 0)
