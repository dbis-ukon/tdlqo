from __future__ import annotations

from typing import List, Optional, Tuple

from schemas.data_types.data_type import DataType
from schemas.data_types.postgresql_data_types import get_postgresql_data_type
from schemas.data_types.string_data_type import StringDataType


class Column:
    def __init__(self,
                 name: str,
                 data_type: DataType,
                 nullable: bool,
                 width: Optional[int] = None,
                 null_fraction: Optional[float] = None,
                 distinct_count: Optional[int] = None,
                 most_common_vals: Optional[List[str]] = None,
                 most_common_freqs: Optional[List[float]] = None,
                 correlation: Optional[float] = None,
                 null_count: Optional[int] = None,
                 max_degree: Optional[int] = None,
                 min_degree: Optional[int] = None):
        self._name = name
        self._data_type = data_type
        self._nullable = nullable
        self._width = width if width is not None else data_type.width()
        self._null_fraction = null_fraction
        self._distinct_count = distinct_count
        self._most_common_vals = most_common_vals
        self._most_common_freqs = most_common_freqs
        self._correlation = correlation
        self._null_count = null_count
        self._max_degree = max_degree
        self._min_degree = min_degree

    def name(self) -> str:
        return self._name

    def data_type(self) -> DataType:
        return self._data_type

    def nullable(self) -> bool:
        return self._nullable

    def width(self) -> int:
        return self._width

    def null_fraction(self) -> Optional[float]:
        return self._null_fraction

    def null_count(self) -> Optional[int]:
        return self._null_count

    def max_degree(self) -> Optional[int]:
        return self._max_degree

    def min_degree(self) -> Optional[int]:
        return self._min_degree

    def distinct_count(self) -> Optional[int]:
        return self._distinct_count

    def mcv_frequency(self) -> Optional[float]:
        if self._most_common_freqs is None:
            return None
        return max(self._most_common_freqs)

    def most_common_vals(self) -> Optional[List[str]]:
        return self._most_common_vals

    def most_common_freqs(self) -> Optional[List[float]]:
        return self._most_common_freqs

    def mcv_items(self) -> Optional[List[Tuple[str, float]]]:
        """Returns list of (value, frequency) pairs, or None if no MCV data."""
        if self._most_common_vals is None or self._most_common_freqs is None:
            return None
        return list(zip(self._most_common_vals, self._most_common_freqs))

    def correlation(self) -> Optional[float]:
        return self._correlation

    @staticmethod
    def build_from_connection(connection, schema_name: str, table_name: str, column_name: str, cardinality: int) -> Column:
        cursor = connection.cursor()
        column_query = """  SELECT isc.data_type, isc.is_nullable = 'YES', isc.numeric_precision, isc.character_maximum_length
                            FROM information_schema.columns AS isc
                            WHERE isc.table_schema = '%s'
                                AND isc.table_name = '%s'
                                AND isc.column_name = '%s';""" % (schema_name, table_name, column_name)
        cursor.execute(column_query)
        data_type_sql, nullable, numeric_precision, maximum_length = cursor.fetchone()
        data_type = get_postgresql_data_type(data_type_sql, numeric_precision=numeric_precision, maximum_length=maximum_length)
        if isinstance(data_type, StringDataType):
            column_query = "EXPLAIN (ANALYZE FALSE, FORMAT JSON) SELECT %s FROM %s;" % (column_name, table_name)
            cursor.execute(column_query)
            width = cursor.fetchone()[0][0]['Plan']['Plan Width']
        else:
            width = None

        stats_query = "SELECT null_frac, n_distinct, most_common_vals::text::text[], most_common_freqs, correlation FROM pg_catalog.pg_stats WHERE schemaname = '%s' AND tablename = '%s' AND attname = '%s';" % (schema_name, table_name, column_name)
        cursor.execute(stats_query)
        stats = cursor.fetchone()
        if stats is not None:
            null_fraction, distinct_count, most_common_vals, mcv_frequencies, correlation = stats
            if distinct_count is not None and distinct_count < 0:
                distinct_count = cardinality * (-distinct_count)
            if mcv_frequencies is None:
                most_common_vals = None
                most_common_freqs = None
            else:
                most_common_freqs = list(mcv_frequencies)
                most_common_vals = list(most_common_vals) if most_common_vals is not None else None
        else:
            null_fraction, distinct_count, correlation = None, None, None
            most_common_vals, most_common_freqs = None, None

        null_count = None
        if not nullable:
            assert null_fraction is None or null_fraction == 0.0
            null_fraction = 0.0
            null_count = 0

        indexed_query = """SELECT COUNT(*) > 0
                                    FROM pg_index ix
                                    JOIN pg_class i ON i.oid = ix.indexrelid
                                    JOIN pg_class t ON t.oid = ix.indrelid
                                    JOIN pg_namespace ns ON ns.oid = t.relnamespace
                                    JOIN pg_attribute a ON a.attrelid = t.oid
                                    WHERE
                                        a.attnum = ANY(ix.indkey)
                                        AND a.attname = '%s'
                                        AND t.relname = '%s';""" % (column_name, table_name)
        cursor.execute(indexed_query)
        is_indexed = cursor.fetchone()[0]

        fk_query = """SELECT COUNT(*) > 0
                        FROM information_schema.key_column_usage kcu
                        JOIN information_schema.table_constraints tc
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                            AND kcu.table_name = '%s'
                            AND kcu.column_name = '%s';""" % (table_name, column_name)
        cursor.execute(fk_query)
        is_fk_referencing = cursor.fetchone()[0]

        exact_distinct = is_indexed or is_fk_referencing

        max_degree = None
        min_degree = None
        if exact_distinct:
            # Combined: distinct count (excl. NULLs) + max/min group size (excl. NULLs).
            # Matches semantics of COUNT(DISTINCT col); NULLs are excluded because NULL never
            # matches in equi-joins, so they don't contribute to degree bounds for joins.
            stats_query = "SELECT COUNT(*), MAX(c), MIN(c) FROM (SELECT COUNT(*) AS c FROM %s WHERE %s IS NOT NULL GROUP BY %s) AS sub;" % (table_name, column_name, column_name)
            cursor.execute(stats_query)
            distinct_count, max_degree, min_degree = cursor.fetchone()

            if nullable and cardinality > 0:
                nulls_query = "SELECT COUNT(*) FROM %s WHERE %s IS NULL;" % (table_name, column_name)
                cursor.execute(nulls_query)
                null_count = cursor.fetchone()[0]
                null_fraction = null_count / cardinality if cardinality > 0 else None

        cursor.close()

        return Column(column_name,
                      data_type,
                      nullable,
                      width=width,
                      null_fraction=null_fraction,
                      distinct_count=distinct_count,
                      most_common_vals=most_common_vals,
                      most_common_freqs=most_common_freqs,
                      correlation=correlation,
                      null_count=null_count,
                      max_degree=max_degree,
                      min_degree=min_degree)
