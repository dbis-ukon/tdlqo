from optimizers.top_down_random_optimizer import TopDownRandomOptimizer
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from queries.spj_query import SPJQuery, SelectClauseElement, SelectClauseAggregationType
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.benchmark_schemas import stack_schema
import random
import numpy as np


def test_determinism():
    query_db = QueryDB("query_db", 5443)
    schema = stack_schema(port=5443)
    benchmark_queries = load_benchmark(query_db, 21)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    available_operators = [HashJoinExpression, NestedLoopJoinExpression, SequentialScanExpression, IndexScanExpression]

    num_trials = 5
    for benchmark_query in benchmark_queries:
        query = benchmark_query.query
        assert isinstance(query, SPJQuery)
        query._select_clause_elements = [SelectClauseElement(None, None, SelectClauseAggregationType.COUNT)]

        plan_strings = []
        for trial in range(num_trials):
            random.seed(0)
            np.random.seed(0)
            optimizer = TopDownRandomOptimizer(available_operators)
            plan = optimizer.optimize(query)
            plan_strings.append(plan.string())

        all_same = all(s == plan_strings[0] for s in plan_strings)
        status = "OK" if all_same else "FAIL"
        print(f"[{status}] {benchmark_query.query_name()}: {len(set(plan_strings))} unique plans out of {num_trials}")
        if not all_same:
            for i, s in enumerate(plan_strings):
                print(f"  trial {i}: {s}")


test_determinism()
