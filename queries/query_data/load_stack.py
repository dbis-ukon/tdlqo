import os
from queries.query_data.query_db import QueryDB


def load_stack(schema_id: int, path: str, query_db: QueryDB):
    benchmark_values = {}
    benchmark_values["schema_id"] = schema_id
    benchmark_values["name"] = "stack"

    benchmark_id = query_db.insert("benchmarks", benchmark_values, return_id=True)

    # 1. Collect all valid SQL files recursively
    found_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check extension and ignored files
            if file.endswith(".sql") and file not in ["schema.sql", "fkindexes.sql"]:
                full_path = os.path.join(root, file)
                # We store the tuple (filename, full_path) to help with sorting and naming later
                found_files.append((file, full_path))

    # 2. Sort the files to ensure deterministic processing order
    # This sorts by filename first. If you prefer sorting by directory structure,
    # you can change this to: found_files.sort(key=lambda x: x[1])
    found_files.sort()

    # 3. Process the files
    for filename, full_path in found_files:
        with open(full_path) as f:
            query_text = f.read()

        # Use the filename (without extension) for the name, as per original code
        query_name = filename[:-4]
        template = full_path.split("/")[-2]

        print(query_name)
        query_text = query_text.strip()
        if not query_text.endswith(";"):
            query_text += ";"
        print(query_text)
        print()

        query_values = {}
        query_values["benchmark_id"] = benchmark_id
        query_values["query"] = query_text
        query_values["name"] = query_name
        query_values["template"] = template
        query_db.insert("benchmark_queries", query_values)

    query_db.commit()




load_stack(3, "/home/silvan/Downloads/stack", QueryDB("query_db", 5443))

