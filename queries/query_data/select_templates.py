

from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from schemas.benchmark_schemas import stack_schema
from schemas.schema import Schema

import random


def roundrobin(*iterables):
    """Yield items from each iterable in turn, skipping exhausted ones."""
    iterators = [iter(it) for it in iterables]
    while iterators:
        for it in iterators[:]:
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)


def select_templates(schema: Schema, query_db: QueryDB, benchmark_id: int, num_queries: int, by_template: bool):
    benchmark_query = "SELECT schema_id, name FROM benchmarks WHERE id = %s;" % benchmark_id
    schema_id, benchmark_name = query_db.search(benchmark_query)[0]

    benchmark_queries = load_benchmark(query_db, benchmark_id)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
    benchmark_queries = [benchmark_query for benchmark_query in benchmark_queries if benchmark_query.query is not None]
    if len(benchmark_queries) <= num_queries:
        print("Not enough valid queries found in benchmark %s. Found %s, required %s." % (benchmark_name, len(benchmark_queries), num_queries))
        return

    rng = random.Random(0)
    rng.shuffle(benchmark_queries)

    benchmark_query_templates = {}
    if by_template:
        for benchmark_query in benchmark_queries:
            if benchmark_query.template not in benchmark_query_templates:
                benchmark_query_templates[benchmark_query.template] = []
            benchmark_query_templates[benchmark_query.template].append(benchmark_query)
    else:
        benchmark_query_templates["all"] = benchmark_queries
    selected_benchmark_queries = []
    for benchmark_query in roundrobin(*[benchmark_query_templates[template] for template in benchmark_query_templates]):
        if len(selected_benchmark_queries) >= num_queries:
            break
        selected_benchmark_queries.append(benchmark_query)

    benchmark_values = {}
    benchmark_values["schema_id"] = schema_id
    benchmark_values["name"] = benchmark_name + "_random_" + ("template_" if by_template else "") + str(num_queries)

    benchmark_id = query_db.insert("benchmarks", benchmark_values, return_id=True)

    for benchmark_query in selected_benchmark_queries:
        query_values = {}
        query_values["benchmark_id"] = benchmark_id
        query_values["query"] = benchmark_query.query_text
        query_values["name"] = benchmark_query.query_name()
        query_values["template"] = benchmark_query.template
        query_db.insert("benchmark_queries", query_values)

    query_db.commit()


schema = stack_schema(port=5443)
query_db = QueryDB("query_db", 5443)
benchmark_id = 17
num_queries = 500
select_templates(schema, query_db, benchmark_id, num_queries, True)

