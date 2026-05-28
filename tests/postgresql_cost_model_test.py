from cardinality_estimators.cardinality_estimator import CardinalityMode
from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from cost_models.main_memory_cost_model import MainMemoryCostModel
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from execution_engines.postgresql_execution_engine import PostgreSQLExecutionEngine
from optimizers.top_down_random_optimizer import TopDownRandomOptimizer
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from queries.spj_query import SPJQuery, SelectClauseElement, SelectClauseAggregationType
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.benchmark_schemas import imdb_schema, stack_schema
import numpy as np
import random


def _node_label(node):
    cls = type(node).__name__.replace("Expression", "")
    if isinstance(node, ScanExpression):
        to = node.table_occurrence
        alias = to.alias() if to.alias() is not None else to.table().name()
        extra = ""
        if isinstance(node, IndexScanExpression):
            idx_name = node.index.name() if node.index is not None else "None"
            extra = f" idx={idx_name} memoized={node.memoized} join_clause={node.join_clause}"
        return f"{cls}({to.table().name()} AS {alias}){extra}"
    return cls


def _collect_local_costs(node, cost_model, acc):
    acc[id(node)] = cost_model.local_cost(node, cardinality_mode=CardinalityMode.MEAN)
    for child in node.children:
        _collect_local_costs(child, cost_model, acc)


def _cumulative_from_local(node, local_by_id):
    local = local_by_id[id(node)]
    if local is None:
        return None
    total = local
    for child in node.children:
        child_total = _cumulative_from_local(child, local_by_id)
        if child_total is None:
            return None
        total += child_total
    return total


def _dump_plan(node, cost_model, cardinality_estimator, depth=0, local_by_id=None):
    if local_by_id is None:
        local_by_id = {}
        _collect_local_costs(node, cost_model, local_by_id)
    local = local_by_id[id(node)]
    cumulative = _cumulative_from_local(node, local_by_id)
    card = cardinality_estimator.estimate(node.canonical_query(), cardinality_mode=CardinalityMode.MEAN)
    card_str = f"{card:.0f}" if card is not None else "None"
    local_str = f"{local:.2f}" if local is not None else "None"
    cum_str = f"{cumulative:.2f}" if cumulative is not None else "None"
    indent = "  " * depth
    print(f"{indent}- {_node_label(node)} local={local_str} cum={cum_str} card={card_str}")
    for child in node.children:
        _dump_plan(child, cost_model, cardinality_estimator, depth=depth + 1, local_by_id=local_by_id)


_PG_PASSTHROUGH_NODE_TYPES = {"Gather", "Gather Merge", "Materialize", "Memoize", "Sort", "Aggregate", "Finalize Aggregate", "Partial Aggregate", "Hash"}


def _pg_node_summary(node):
    nt = node.get("Node Type", "?")
    alias = node.get("Alias")
    relation = node.get("Relation Name")
    index = node.get("Index Name")
    parts = [nt]
    if relation is not None:
        parts.append(f"rel={relation}")
    if alias is not None and alias != relation:
        parts.append(f"as={alias}")
    if index is not None:
        parts.append(f"idx={index}")
    if "Inner Unique" in node:
        parts.append(f"inner_unique={node['Inner Unique']}")
    return " ".join(parts)


def _dump_pg_plan(node, depth=0):
    startup = node.get("Startup Cost", 0.0)
    total = node.get("Total Cost", 0.0)
    rows = node.get("Plan Rows", 0)
    children = node.get("Plans", []) or []
    children_total = sum(child.get("Total Cost", 0.0) for child in children)
    local = total - children_total
    passthrough = " [passthrough]" if node.get("Node Type") in _PG_PASSTHROUGH_NODE_TYPES else ""
    indent = "  " * depth
    print(f"{indent}- {_pg_node_summary(node)} local={local:.2f} startup={startup:.2f} total={total:.2f} rows={rows}{passthrough}")
    for key in ("Hash Cond", "Index Cond", "Recheck Cond", "Join Filter", "Filter"):
        if key in node:
            print(f"{indent}    {key}: {node[key]}")
    for child in children:
        _dump_pg_plan(child, depth=depth + 1)


def postgresql_cost_model_test():
    query_db = QueryDB("query_db", 5443)
    schema = stack_schema(port=5443)
    benchmark_queries = load_benchmark(query_db, 20)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    available_operators = [HashJoinExpression, NestedLoopJoinExpression, SequentialScanExpression, IndexScanExpression]
    optimizer = TopDownRandomOptimizer(available_operators)
    execution_engine = PgHintPlanExecutionEngine(schema, 0)

    cursor = schema.connection().cursor()
    for table in schema.tables():
        estimated_width = table.width()
        table_query = "EXPLAIN (ANALYZE FALSE, FORMAT JSON) SELECT * FROM %s;" % table.name()
        cursor.execute(table_query)
        actual_width = cursor.fetchone()[0][0]['Plan']['Plan Width']
        print("Table %s: Estimated width: %d, PostgreSQL width: %d" % (table.name(), estimated_width, actual_width))

    print()

    for table in schema.tables():
        for column in table.columns():
            estimated_width = column.width()
            column_query = "EXPLAIN (ANALYZE FALSE, FORMAT JSON) SELECT %s FROM %s;" % (column.name(), table.name())
            cursor.execute(column_query)
            actual_width = cursor.fetchone()[0][0]['Plan']['Plan Width']
            print("Column %s.%s: Estimated width: %d, PostgreSQL width: %d" % (table.name(), column.name(), estimated_width, actual_width))

    cursor.close()
    print()

    q_errors = []
    ratios = []
    estimated_costs = []
    postgresql_costs = []
    for repeat_idx in range(1):
        for benchmark_query in benchmark_queries:
            seed = hash((benchmark_query.query_id, repeat_idx)) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            query = benchmark_query.query
            assert isinstance(query, SPJQuery)
            query._select_clause_elements = [SelectClauseElement(None, None, SelectClauseAggregationType.COUNT)]
            plan = optimizer.optimize(query)
            execution_data = execution_engine.execute(query, plan, analyze=False, collect_cardinality_estimates=True)
            postgresql_cardinality_estimator = PostgreSQLCardinalityEstimator(schema, cache=True)
            postgresql_cost_model = PostgreSQLCostModel(postgresql_cardinality_estimator, schema.postgresql_configuration(), default_width=8)
            for query, cardinality_estimate in execution_data.cardinality_estimates.items():
                postgresql_cardinality_estimator.memorize(query, cardinality_estimate)
            estimated_cost = postgresql_cost_model.cost(plan)
            join_estimated_cost = postgresql_cost_model.operator_type_cost(plan, JoinExpression)
            scan_estimated_cost = postgresql_cost_model.operator_type_cost(plan, ScanExpression)
            postgresql_cost = execution_data.estimated_cost
            if postgresql_cost is None:
                print("Query %s timed out" % benchmark_query.query_name())
            elif not execution_data.executed_as_intended:
                print("Query %s did not execute as intended" % benchmark_query.query_name())
            else:
                ratio = estimated_cost / postgresql_cost
                ratios.append(ratio)
                q_error = max(ratio, 1 / ratio)
                q_errors.append(q_error)
                estimated_costs.append(estimated_cost)
                postgresql_costs.append(postgresql_cost)
                hash_cost = postgresql_cost_model.operator_type_cost(plan, HashJoinExpression)
                nl_cost = postgresql_cost_model.operator_type_cost(plan, NestedLoopJoinExpression)
                seq_cost = postgresql_cost_model.operator_type_cost(plan, SequentialScanExpression)
                idx_cost = postgresql_cost_model.operator_type_cost(plan, IndexScanExpression)
                direction = "OVER" if ratio > 1 else "UNDER"
                print("Query %s: Estimated Cost: %.0f, PostgreSQL Cost: %.0f, ratio: %.4f (%s q-error=%.2f), join cost: %.0f, scan cost: %.0f" % (benchmark_query.query_name(), estimated_cost, postgresql_cost, ratio, direction, q_error, join_estimated_cost, scan_estimated_cost))
                print(f"  breakdown: hash={hash_cost:.0f} nl={nl_cost:.0f} seq={seq_cost:.0f} idx={idx_cost:.0f}")
                print(f"  plan string: {plan.string()}")
                print("  our plan tree:")
                postgresql_cost_model.debug_hash_join = True
                postgresql_cost_model.debug_nested_loop = True
                _dump_plan(plan, postgresql_cost_model, postgresql_cardinality_estimator, depth=2)
                postgresql_cost_model.debug_hash_join = False
                postgresql_cost_model.debug_nested_loop = False
                if execution_data.explain_json is not None:
                    print("  postgresql plan tree:")
                    _dump_pg_plan(execution_data.explain_json["Plan"], depth=2)
                print()
    print()
    print("Geometric mean q-error: %.4f" % np.exp(np.mean(np.log(q_errors))))
    print("Mean q-error: %.4f" % np.mean(q_errors))
    print("Median q-error: %.4f" % np.median(q_errors))
    print()
    print("Geometric mean ratio: %.4f" % np.exp(np.mean(np.log(ratios))))
    print("Mean ratio: %.4f" % np.mean(ratios))
    print("Median ratio: %.4f" % np.median(ratios))
    print()
    estimated_ranks = np.argsort(np.argsort(estimated_costs))
    postgresql_ranks = np.argsort(np.argsort(postgresql_costs))
    spearman = np.corrcoef(estimated_ranks, postgresql_ranks)[0, 1]
    print("Spearman rank correlation (pooled across plans): %.4f" % spearman)



postgresql_cost_model_test()







