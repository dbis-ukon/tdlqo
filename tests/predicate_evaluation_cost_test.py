
import numpy as np
import prettytable

from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from queries.benchmark_query import BenchmarkQuery
from queries.query_data.parse import parse_spj_query
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.benchmark_schemas import imdb_schema


def predicate_evaluation_cost_test(repetitions = 10):
    base_query = "SELECT COUNT(*) FROM cast_info AS ci%s;"
    predicates = ["ci.note IS NULL",
                  "ci.note IS NOT NULL",
                  "ci.nr_order = 1",
                  "ci.nr_order > 1",
                  "ci.nr_order != 1",
                  "ci.nr_order = 1 OR ci.nr_order = 3",
                  "ci.nr_order IN (1, 3)",
                  "ci.nr_order = 1 OR ci.nr_order = 3 OR ci.nr_order = 5",
                  "ci.nr_order IN (1, 3, 5)",
                  "ci.note = '(uncredited)'",
                  "ci.note LIKE '%(uncredited)%'",
                  "ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%'",
                  "ci.note LIKE '%(voice)%' OR ci.note LIKE '%(uncredited)%'"]
    queries = [base_query % ""]
    for predicate in predicates:
        query = base_query % (" WHERE " + predicate)
        queries.append(query)

    schema = imdb_schema(port=5443)
    cursor = schema.connection().cursor()
    cursor.execute("SET max_parallel_workers_per_gather = 0;")

    cardinality_estimator = PropheticCardinalityEstimator()
    cost_model = PostgreSQLCostModel(cardinality_estimator, schema.postgresql_configuration(), improved=True)

    table = prettytable.PrettyTable()
    table.field_names = ["Query", "Actual Cardinality", "PostgreSQL Cost", "Reimplementation Cost", "PostgreSQL Execution Time (ms)"]
    for query in queries:
        cursor.execute(query)
        benchmark_query = BenchmarkQuery(0, query)
        spj_query = parse_spj_query(schema, benchmark_query, verify_correctness=False)
        table_occurrence = list(spj_query.table_occurrences())[0]
        sequential_scan = SequentialScanExpression(table_occurrence)
        reimplementation_cost = cost_model.cost(sequential_scan)

        actual_cardinality = cursor.fetchone()[0]
        explain_query = "EXPLAIN (ANALYZE TRUE, VERBOSE TRUE, FORMAT JSON) " + query
        postgresql_costs = []
        execution_times = []
        print("Executing query:", query)
        for _ in range(repetitions):
            cursor.execute(explain_query)
            explain_json = cursor.fetchone()[0]
            plan = explain_json[0]['Plan']
            total_cost = plan['Total Cost']
            execution_time = explain_json[0]['Execution Time']
            postgresql_costs.append(total_cost)
            execution_times.append(execution_time)
        mean_cost = np.mean(postgresql_costs)
        mean_execution_time = np.mean(execution_times)
        table.add_row([query, actual_cardinality, f"{mean_cost:.2f}", f"{reimplementation_cost:.2f}", f"{mean_execution_time:.2f}"])

    cursor.close()
    table.sortby = "Actual Cardinality"
    print(table)




predicate_evaluation_cost_test()








