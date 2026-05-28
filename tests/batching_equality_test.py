import statistics
from typing import List, Tuple

import torch
import torch_geometric
import random

torch.use_deterministic_algorithms(True)

from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import \
    PlanType
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_encoder import TopDownLearnedOptimizerEncoder
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_module import TopDownLearnedOptimizerModule
from optimizers.top_down_optimizer import TopDownOptimizer
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from queries.spj_query import SPJQuery
from relational_algebra_expressions.group_relational_algebra_expression import GroupRelationalAlgebraExpression
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.requirements import Requirements
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.benchmark_schemas import imdb_schema, stack_schema


def _record(records: List[Tuple[float, float]], a: float, b: float, label: str) -> None:
    abs_diff = abs(a - b)
    denom = max(abs(a), abs(b))
    rel_diff = abs_diff / denom if denom > 0 else 0.0
    records.append((abs_diff, rel_diff))
    print("  %-32s a=%+.9e  b=%+.9e  abs=%.3e  rel=%.3e" % (label, a, b, abs_diff, rel_diff))


def _print_summary(records: List[Tuple[float, float]], header: str) -> None:
    print()
    print("=== %s ===" % header)
    n = len(records)
    if n == 0:
        print("  no comparisons recorded")
        return
    abs_diffs = [r[0] for r in records]
    rel_diffs = [r[1] for r in records]
    print("  N = %d" % n)
    print("  abs_diff   min=%.3e  median=%.3e  mean=%.3e  max=%.3e" % (
        min(abs_diffs), statistics.median(abs_diffs), sum(abs_diffs) / n, max(abs_diffs)))
    print("  rel_diff   min=%.3e  median=%.3e  mean=%.3e  max=%.3e" % (
        min(rel_diffs), statistics.median(rel_diffs), sum(rel_diffs) / n, max(rel_diffs)))
    for threshold in (1e-9, 1e-6, 1e-3, 1e-1):
        over_abs = sum(1 for d in abs_diffs if d > threshold)
        over_rel = sum(1 for d in rel_diffs if d > threshold)
        print("  abs_diff > %.0e: %d/%d (%.1f%%)   rel_diff > %.0e: %d/%d (%.1f%%)" % (
            threshold, over_abs, n, 100.0 * over_abs / n,
            threshold, over_rel, n, 100.0 * over_rel / n))


def batching_equality_test():
    schema = imdb_schema(port=5443)
    configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
    encoder = TopDownLearnedOptimizerEncoder(configuration, schema)
    module = TopDownLearnedOptimizerModule(configuration, encoder)
    available_operators = [HashJoinExpression, SequentialScanExpression]
    optimizer = TopDownOptimizer("test", "test", "test", available_operators)

    query_db = QueryDB("query_db", 5443)
    benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    query_datas = []
    join_outputs = []
    for benchmark_query in benchmark_queries:
        query = benchmark_query.query
        assert isinstance(query, SPJQuery)
        group = GroupRelationalAlgebraExpression(query, Requirements())
        possible_plans = optimizer.enumerate_memoized_expressions(group)
        query_data, plan_dict, _ = encoder.encode(group, possible_plans)
        query_datas.append(query_data)
        join_output, _ = module.forward(query_data)
        join_outputs.append(join_output)

    join_outputs_tensor = torch.cat(join_outputs, dim=0)

    loader = torch_geometric.loader.DataLoader(query_datas, batch_size=len(query_datas))
    unified_query_data = next(iter(loader))

    unified_join_output, _ = module.forward(unified_query_data)

    records: List[Tuple[float, float]] = []
    print("--- batching_equality_test: per-plan comparisons (individual vs batched) ---")
    for i, (output, unified_output) in enumerate(zip(join_outputs_tensor, unified_join_output)):
        _record(records, output.item(), unified_output.item(), label="plan %d" % i)
    _print_summary(records, header="batching_equality_test summary")


def plan_permutation_equality_test():
    schema = imdb_schema(port=5443)
    configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
    encoder = TopDownLearnedOptimizerEncoder(configuration, schema)
    module = TopDownLearnedOptimizerModule(configuration, encoder)
    available_operators = [HashJoinExpression, SequentialScanExpression]
    optimizer = TopDownOptimizer("test", "test", "test", available_operators)

    query_db = QueryDB("query_db", 5443)
    benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    overall_records: List[Tuple[float, float]] = []
    for benchmark_query in benchmark_queries:
        query = benchmark_query.query
        assert isinstance(query, SPJQuery)
        group = GroupRelationalAlgebraExpression(query, Requirements())
        possible_plans = optimizer.enumerate_memoized_expressions(group)
        query_data, plan_dict, _ = encoder.encode(group, possible_plans)
        join_output, _ = module.forward(query_data)
        permuted_possible_plans = possible_plans.copy()
        random.shuffle(permuted_possible_plans)
        permuted_query_data, permuted_plan_dict, _ = encoder.encode(group, permuted_possible_plans)
        permuted_join_output, _ = module.forward(permuted_query_data)
        output_dict = {}
        permuted_output_dict = {}
        for i, (output_value, permuted_output_value) in enumerate(zip(join_output, permuted_join_output)):
            plan = plan_dict[(PlanType.JOIN, i)]
            permuted_plan = permuted_plan_dict[(PlanType.JOIN, i)]
            output_dict[plan] = output_value
            permuted_output_dict[permuted_plan] = permuted_output_value
        per_query_records: List[Tuple[float, float]] = []
        print("--- plan_permutation_equality_test: %s (%d plans) ---" % (benchmark_query.query_name(), len(possible_plans)))
        for plan in possible_plans:
            output_value = output_dict[plan]
            permuted_output_value = permuted_output_dict[plan]
            _record(per_query_records, output_value.item(), permuted_output_value.item(), label=plan.string()[:60])
        _print_summary(per_query_records, header="%s summary" % benchmark_query.query_name())
        overall_records.extend(per_query_records)
    _print_summary(overall_records, header="plan_permutation_equality_test overall summary")




batching_equality_test()
print()
plan_permutation_equality_test()
