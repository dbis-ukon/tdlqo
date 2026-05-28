from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from sample_encoders.table_row import TableRow
from schemas.benchmark_schemas import imdb_schema


def predicate_evaluation_test(limit: int = 1000):
    schema = imdb_schema(port=5443)
    query_db = QueryDB("query_db", 5443)
    benchmark_queries = load_benchmark(query_db, 15)
    parse_spj_queries(schema, benchmark_queries, verify_correctness=False)

    cursor = schema.connection().cursor()
    cursor.execute("SET enable_indexscan = off;")
    cursor.execute("SET enable_bitmapscan = off;")
    samples = {}
    for table in schema.tables():
        columns = list(table.columns())
        column_string = ", ".join([column.name() for column in columns])
        sample_query = "SELECT %s FROM %s ORDER BY ctid LIMIT %d;" % (column_string, table.name(), limit)
        cursor.execute(sample_query)
        table_samples = []
        for postgresql_tuple in cursor.fetchall():
            table_samples.append(TableRow.build_table_row(columns, postgresql_tuple))
        samples[table] = table_samples

    for benchmark_query in benchmark_queries:
        for table_occurrence in benchmark_query.query.table_occurrences():
            table = table_occurrence.table()
            predicate = table_occurrence.predicate()

            evaluation_query = "SELECT %s FROM %s AS %s ORDER BY ctid LIMIT %d;" % (predicate.alias_string(), table.name(), table_occurrence.alias(), limit)
            cursor.execute(evaluation_query)
            sql_evaluation = []
            for postgresql_tuple in cursor.fetchall():
                sql_evaluation.append(postgresql_tuple[0])

            for sample, sql_result in zip(samples[table], sql_evaluation):
                python_result = predicate.evaluate(sample)
                if python_result != sql_result:
                    raise ValueError("Mismatch in predicate evaluation for predicate (%s) on table %s: Python result %s, SQL result %s" % (predicate.alias_string(), table.name(), str(python_result), str(sql_result)))

            print("Predicate evaluation match for predicate (%s) on table %s" % (predicate.alias_string(), table.name()))

    cursor.close()




predicate_evaluation_test()




