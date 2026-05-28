

CREATE TABLE databases(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE schemas(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    database_id INT NOT NULL REFERENCES databases(id)
);

CREATE TABLE tables(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    schema_id INT NOT NULL REFERENCES schemas(id)
);

CREATE TABLE benchmarks(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    schema_id INT NOT NULL REFERENCES schemas(id)
);


CREATE TABLE spj_queries(
    id SERIAL PRIMARY KEY
);

CREATE TABLE table_occurrences(
    id SERIAL PRIMARY KEY,
    schema_table_id INT NOT NULL REFERENCES tables(id),
    alias VARCHAR(255),
    spj_query_id INT NOT NULL REFERENCES spj_queries(id)
);

CREATE TABLE predicates(
    id SERIAL PRIMARY KEY,
    predicate TEXT NOT NULL
);

CREATE TABLE predicate_table_occurrences(
    predicate_id INT NOT NULL REFERENCES predicates(id),
    position INT NOT NULL,
    table_occurrence_id INT NOT NULL REFERENCES table_occurrences(id),
    PRIMARY KEY(predicate_id, position)
);

CREATE TABLE joins(
    id SERIAL PRIMARY KEY,
    spj_query_id INT NOT NULL REFERENCES spj_queries(id)
);

CREATE TABLE join_participants(
    equi_join_id INT NOT NULL REFERENCES joins(id),
    table_occurrence_id INT NOT NULL REFERENCES table_occurrences(id),
    column_name VARCHAR(255) NOT NULL,
    PRIMARY KEY(equi_join_id, table_occurrence_id, column_name)
);

CREATE TABLE benchmark_queries(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    query TEXT NOT NULL,
    benchmark_id INT NOT NULL REFERENCES benchmarks(id),
    spj_query_id INT REFERENCES spj_queries(id)
);





