import os
from queries.query_data.query_db import QueryDB


def load_stats_ceb(schema_id: int, filepath: str, query_db: QueryDB):
    # 1. Register the new benchmark
    benchmark_values = {}
    benchmark_values["schema_id"] = schema_id
    benchmark_values["name"] = "STATS-CEB"

    benchmark_id = query_db.insert("benchmarks", benchmark_values, return_id=True)

    # 2. Read the single file containing all queries
    with open(filepath, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} queries in {filepath}")

    # 3. Process each line
    for i, line in enumerate(lines):
        if not line.strip():
            continue  # Skip empty lines

        # Split the line: "79851||SELECT..." -> ["79851", "SELECT..."]
        parts = line.split("||")

        if len(parts) != 2:
            raise ValueError(f"Line {i+1} is malformed: {line}")

        # The first part is the cardinality (label), the second is the SQL
        # If you want to store the cardinality, you would need a column for it.
        # Here we strictly extract the SQL.
        query_text = parts[1].strip()
        if not query_text.endswith(";"):
            query_text += ";"

        # Generate a unique name for the query based on its order
        query_name = f"query_{i}"

        print(f"{query_name}: {query_text}")

        query_values = {}
        query_values["benchmark_id"] = benchmark_id
        query_values["query"] = query_text
        query_values["name"] = query_name
        query_db.insert("benchmark_queries", query_values)

    query_db.commit()


load_stats_ceb(2, "/home/silvan/code/stats_CEB.sql", QueryDB("query_db", 5443))



