
import os
from queries.query_data.query_db import QueryDB


def load_job(schema_id: int, path: str, query_db: QueryDB):
    benchmark_values = {}
    benchmark_values["schema_id"] = schema_id
    benchmark_values["name"] = "JOB"

    benchmark_id = query_db.insert("benchmarks", benchmark_values, return_id=True)

    files = []
    for file in os.listdir(path):
        if file.endswith(".sql") and file not in ["schema.sql", "fkindexes.sql"]:
            files.append(file)
    files.sort()
    for file in files:
        with open(os.path.join(path, file)) as f:
            query_text = f.read()
        print(file[:-4])
        query_text = query_text.strip()
        print(query_text)
        print()
        query_values = {}
        query_values["benchmark_id"] = benchmark_id
        query_values["query"] = query_text
        query_values["name"] = file[:-4]
        query_db.insert("benchmark_queries", query_values)

    query_db.commit()




# load_job(1, "/home/silvan/Downloads/join-order-benchmark-master", QueryDB("query_db", 5440))
















