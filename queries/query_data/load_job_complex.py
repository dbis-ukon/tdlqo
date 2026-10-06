from queries.query_data.query_db import QueryDB


def load_job_complex(schema_id: int, filepath: str, query_db: QueryDB):
    # 1. Register the new benchmark
    benchmark_values = {}
    benchmark_values["schema_id"] = schema_id
    benchmark_values["name"] = "JOB-Complex"

    benchmark_id = query_db.insert("benchmarks", benchmark_values, return_id=True)

    # 2. Read the single file containing all queries (one query per line)
    with open(filepath) as f:
        lines = f.readlines()

    print(f"Found {len(lines)} lines in {filepath}")

    # 3. Process each line
    query_index = 0
    for line in lines:
        query_text = line.strip()
        if not query_text:
            continue  # Skip empty lines

        if not query_text.endswith(";"):
            query_text += ";"

        query_index += 1
        query_name = f"{query_index}"

        print(query_name)
        print(query_text)
        print()

        query_values = {}
        query_values["benchmark_id"] = benchmark_id
        query_values["query"] = query_text
        query_values["name"] = query_name
        query_db.insert("benchmark_queries", query_values)

    query_db.commit()


# Example usage
# load_job_complex(1, "/home/silvan/Downloads/JOB-Complex.sql", QueryDB("query_db", 5443))
