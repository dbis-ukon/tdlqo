

from __future__ import annotations

import random
from typing import List, Dict, Optional, Iterable, Tuple

import torch
import json
import numpy as np

from queries.predicates.true_predicate import TruePredicate
from queries.spj_query import SPJQuery
from queries.table_occurrence import TableOccurrence
from sample_encoders.greedy_feature_selection import greedy_feature_selection
from schemas.schema import Schema
from schemas.table import Table
from sample_encoders.table_row import TableRow


class SampleEncoder:
    def __init__(self, schema: Schema, samples: Dict[Table, List[TableRow]], max_sample_size: int):
        self._schema = schema
        self._samples = samples
        self._max_sample_size = max_sample_size

    def max_sample_size(self) -> int:
        return self._max_sample_size

    def encode(self, table_occurrence: TableOccurrence) -> torch.FloatTensor:
        sample_bitmap = torch.zeros(self._max_sample_size, dtype=torch.float32)
        for i, table_row in enumerate(self._samples[table_occurrence.table()]):
            sample_bitmap[i] = 1.0 if table_occurrence.predicate().evaluate(table_row) else 0.0
        return sample_bitmap

    @staticmethod
    def empty_sample_encoder(schema: Schema, max_sample_size: int) -> SampleEncoder:
        samples = {}
        for table in schema.tables():
            samples[table] = []
        return SampleEncoder(schema, samples, max_sample_size)

    @staticmethod
    def random_sample_encoder(schema: Schema, sample_size: int) -> SampleEncoder:
        connection = schema.connection()
        cursor = connection.cursor()

        table_samples = {}

        for table in schema.tables():
            table_samples[table] = get_samples_sql(cursor, table, sample_size)

        cursor.close()

        samples = table_samples
        return SampleEncoder(schema, samples, sample_size)

    @staticmethod
    def feature_selected_sample_encoder(schema: Schema, max_sample_size: int, training_queries: Iterable[SPJQuery], max_fs_choices: int = 1000) -> SampleEncoder:
        samples, _ = SampleEncoder._feature_selection_base(schema, max_sample_size, training_queries, max_fs_choices, None)
        return SampleEncoder(schema, samples, max_sample_size)

    def update_feature_selected_sample_encoder(self, training_queries: Iterable[SPJQuery], max_fs_choices: int = 1000) -> bool:
        self._samples, changed = SampleEncoder._feature_selection_base(self._schema, self._max_sample_size, training_queries, max_fs_choices, self._samples)
        return changed

    @staticmethod
    def _feature_selection_base(schema: Schema, max_sample_size: int, training_queries: Iterable[SPJQuery], max_fs_choices: int, old_samples: Optional[Dict[Table, List[TableRow]]]) -> Tuple[Dict[Table, List[TableRow]], bool]:
        table_occurrences = {}
        for table in schema.tables():
            base_table_occurrence = TableOccurrence(table)
            base_table_occurrence.set_predicate(TruePredicate())
            table_occurrences[table] = {base_table_occurrence}

        for query in training_queries:
            for table_occurrence in query.table_occurrences():
                if not isinstance(table_occurrence.predicate(), TruePredicate):
                    table_occurrences[table_occurrence.table()].add(table_occurrence)

        changed = False

        cursor = schema.connection().cursor()
        samples = {}
        for table in table_occurrences:
            current_table_occurrences = table_occurrences[table]
            if len(current_table_occurrences) > max_fs_choices:
                current_table_occurrences = random.sample(list(current_table_occurrences), max_fs_choices)
            if len(current_table_occurrences) <= 1:
                samples[table] = []
                print("%s: %d samples" % (table.name(), len(samples[table])))
                continue
            if old_samples is not None and len(old_samples[table]) >= max_sample_size:
                samples[table] = old_samples[table]
                print("%s: %d samples" % (table.name(), len(samples[table])))
                continue
            a_set = set()
            a_list = []
            candidate_dict = {}
            if old_samples is not None:
                for old_sample in old_samples[table]:
                    a = np.zeros(len(current_table_occurrences))
                    for j, table_occurrence in enumerate(current_table_occurrences):
                        a[j] = table_occurrence.predicate().evaluate(old_sample)
                    a_tuple = tuple(a)
                    a_set.add(a_tuple)
                    candidate_dict[len(a_list)] = old_sample
                    a_list.append(a)
                old_sample_length = len(old_samples[table])
            else:
                old_sample_length = 0
            candidates = get_samples_sql(cursor, table, max_fs_choices)
            if len(candidates) <= old_sample_length and old_samples is not None:
                assert len(candidates) == old_sample_length
                samples[table] = old_samples[table]
                print("%s: %d samples" % (table.name(), len(samples[table])))
                continue
            for i, candidate in enumerate(candidates):
                a = np.zeros(len(current_table_occurrences))
                for j, table_occurrence in enumerate(current_table_occurrences):
                    a[j] = table_occurrence.predicate().evaluate(candidate) == True  # == True is necessary for None values
                # check if a is all zeros or all ones
                a_sum = np.sum(a)
                if a_sum == 0 or a_sum == len(a):
                    continue
                a_tuple = tuple(a)
                if a_tuple not in a_set:
                    a_set.add(a_tuple)
                    candidate_dict[len(a_list)] = candidate
                    a_list.append(a)
                    if len(a_list) >= max_fs_choices:
                        break
            if len(a_set) == 0:
                samples[table] = []
            else:
                A = np.array(a_list).T
                sample_ids = greedy_feature_selection(A, min(max_sample_size, len(a_list)), old_sample_length=old_sample_length)
                samples[table] = [candidate_dict[i] for i in sample_ids]
            sample_length = len(samples[table])
            print("%s: %d samples" % (table.name(), sample_length))
            if sample_length > old_sample_length:
                changed = True
        cursor.close()
        return samples, changed

    def save(self, path: str):
        sample_row_dict = {}
        for table in self._samples:
            sample_row_dict[table.name()] = []
            primary_key_columns = list(table.primary_key().columns())
            primary_key_columns.sort(key=lambda column: column.name())
            for table_row in self._samples[table]:
                pk_values = tuple(table_row.values[column] for column in primary_key_columns)
                sample_row_dict[table.name()].append(pk_values)
        with open(path, "w") as f:
            json.dump(sample_row_dict, f, default=str)

    def load(self, path: str):
        with open(path, "r") as f:
            sample_row_dict = json.load(f)
        samples = {}
        connection = self._schema.connection()
        cursor = connection.cursor()
        for table in self._samples:
            samples[table] = []
            primary_key_columns = list(table.primary_key().columns())
            primary_key_columns.sort(key=lambda column: column.name())
            column_string = ", ".join([column.name() for column in table.columns()])
            for pk_values in sample_row_dict[table.name()]:
                where_clauses = []
                for column, value in zip(primary_key_columns, pk_values):
                    if value is None:
                        where_clauses.append("%s IS NULL" % column.name())
                    elif isinstance(value, str):
                        where_clauses.append("%s = '%s'" % (column.name(), value.replace("'", "''")))
                    else:
                        where_clauses.append("%s = %s" % (column.name(), str(value)))
                where_clause = " AND ".join(where_clauses)
                sample_query = "SELECT %s FROM %s WHERE %s;" % (column_string, table.name(), where_clause)
                cursor.execute(sample_query)
                postgresql_tuple = cursor.fetchone()
                if postgresql_tuple is not None:
                    table_row = TableRow.build_table_row(list(table.columns()), postgresql_tuple)
                    samples[table].append(table_row)
        cursor.close()
        self._samples = samples


def get_samples_sql(cursor, table: Table, sample_size: int) -> List[TableRow]:
    columns = list(table.columns())
    column_string = ", ".join([column.name() for column in columns])
    if table.cardinality() <= sample_size:
        sample_query = "SELECT %s FROM %s;" % (column_string, table.name())
    else:
        sample_query = "SELECT %s FROM %s ORDER BY RANDOM() LIMIT %d;" % (column_string, table.name(), sample_size)
    cursor.execute(sample_query)
    table_samples = []
    for postgresql_tuple in cursor.fetchall():
        table_samples.append(TableRow.build_table_row(columns, postgresql_tuple))
    return table_samples



