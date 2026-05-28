from typing import List

from queries.benchmark_query import BenchmarkQuery
from queries.query_data.query_db import QueryDB


def load_benchmark(query_db: QueryDB, benchmark_id: int) -> List[BenchmarkQuery]:
    query_query = """SELECT id, name, query, template FROM benchmark_queries WHERE benchmark_id = %d ORDER BY id;""" % benchmark_id
    queries = query_db.search(query_query)
    benchmark_queries = []
    for query_id, query_name, query_text, template in queries:
        benchmark_query = BenchmarkQuery(query_id, query_text, query_name=query_name, template=template)
        benchmark_queries.append(benchmark_query)
    return benchmark_queries



# load_benchmark(QueryDB("query_db", 5440), 2)


