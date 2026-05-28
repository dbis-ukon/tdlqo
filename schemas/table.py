from typing import FrozenSet, List, Tuple

from schemas.column import Column
from schemas.index import Index


class Table:
    def __init__(self,
                 oid: int,
                 name: str,
                 columns: List[Column],
                 primary_key: Index,
                 non_pk_indexes: List[Index],
                 cardinality: int,
                 table_size: int):
        self._oid = oid
        self._name = name
        self._columns = frozenset(columns)
        self._column_dict = {column.name(): column for column in columns}
        self._primary_key = primary_key
        indexes = [primary_key] + non_pk_indexes
        for index in indexes:
            for column in index.columns():
                assert(column in self._columns)
        self._indexes = frozenset(indexes)
        self._index_dict = {index.name(): index for index in indexes}
        self._cardinality = cardinality
        self._table_size = table_size

    def name(self) -> str:
        return self._name

    def columns(self) -> FrozenSet[Column]:
        return self._columns

    def column(self, name: str) -> Column:
        assert(name in self._column_dict)
        return self._column_dict[name]

    def width(self) -> int:
        return sum(column.width() for column in self._columns)

    def primary_key(self) -> Index:
        return self._primary_key

    def indexes(self) -> FrozenSet[Index]:
        return self._indexes

    def index(self, name: str) -> Index:
        assert(name in self._index_dict)
        return self._index_dict[name]

    def cardinality(self) -> int:
        return self._cardinality

    def table_size(self) -> int:
        return self._table_size

    @staticmethod
    def build_from_connection(connection,
                              name: str,
                              schema: str):
        cardinality = Table._get_cardinality(connection, name)
        table_size = Table._get_table_size(connection, name)
        columns = Table._get_columns(connection, schema, name, cardinality)
        oid = Table._get_oid(connection, name)
        primary_key, indexes = Table._get_indexes(connection, schema, name, columns, cardinality)
        return Table(oid, name, columns, primary_key, indexes, cardinality, table_size)

    @staticmethod
    def _get_cardinality(connection, name: str) -> int:
        cursor = connection.cursor()
        cardinality_query = "SELECT COUNT(*) FROM %s;" % name
        cursor.execute(cardinality_query)
        cardinality = cursor.fetchone()[0]
        cursor.close()
        return cardinality

    @staticmethod
    def _get_table_size(connection, name: str) -> int:
        cursor = connection.cursor()
        size_query = "SELECT pg_table_size('%s');" % name
        cursor.execute(size_query)
        table_size = cursor.fetchone()[0]
        cursor.close()
        return table_size

    @staticmethod
    def _get_columns(connection, schema: str, table_name: str, cardinality: int) -> List[Column]:
        cursor = connection.cursor()
        column_query = """  SELECT c.column_name
                            FROM information_schema.columns AS c
                            WHERE c.table_schema = '%s'
                                AND c.table_name = '%s';""" % (schema, table_name)
        cursor.execute(column_query)
        columns = []
        for column_name, in cursor.fetchall():
            attribute = Column.build_from_connection(connection, schema, table_name, column_name, cardinality)
            columns.append(attribute)
        return columns

    @staticmethod
    def _get_oid(connection, name: str) -> int:
        cursor = connection.cursor()
        oid_query = "SELECT oid FROM pg_class WHERE relkind = 'r' AND relname = '%s';" % name
        cursor.execute(oid_query)
        oid = cursor.fetchone()[0]
        cursor.close()
        return oid

    @staticmethod
    def _get_indexes(connection, schema_name: str, table_name: str, columns: List[Column], cardinality: int) -> Tuple[Index, List[Index]]:
        column_dict = {column.name(): column for column in columns}
        cursor = connection.cursor()
        index_query = """SELECT pci.oid, pci.relname, pi.indisprimary, pi.indisunique, pci.relpages, pci.reltuples
                        FROM pg_catalog.pg_index pi
                        JOIN pg_catalog.pg_class pci ON pci.oid = pi.indexrelid
                        JOIN pg_catalog.pg_class pct ON pct.oid = pi.indrelid
                        JOIN pg_catalog.pg_namespace pns ON pns.oid = pci.relnamespace
                        WHERE pns.nspname = '%s'
                        AND pct.relname = '%s'
                        ORDER BY pci.relname;""" % (schema_name, table_name)
        cursor.execute(index_query)
        index_names = {}
        index_unique = {}
        index_pages = {}
        index_tuples = {}
        primary_key_id = None
        for oid, index_name, is_primary_key, unique, pages, tuples in cursor.fetchall():
            if is_primary_key:
                assert(primary_key_id is None)
                primary_key_id = oid
            index_names[oid] = index_name
            index_unique[oid] = unique
            index_pages[oid] = pages
            index_tuples[oid] = tuples
        assert(primary_key_id is not None)
        column_query = """SELECT pci.oid, idx.index, pa.attname
                        FROM pg_catalog.pg_index pi
                        JOIN pg_catalog.pg_class pci ON pci.oid = pi.indexrelid
                        JOIN pg_catalog.pg_class pct ON pct.oid = pi.indrelid
                        JOIN pg_catalog.pg_attribute pa ON pa.attrelid = pct.oid AND pa.attnum = ANY(string_to_array(pi.indkey::text, ' ')::int2[])
                        JOIN pg_catalog.pg_namespace pns ON pns.oid = pci.relnamespace
                        CROSS JOIN LATERAL (
                            SELECT ordinality as index
                            FROM unnest(string_to_array(pi.indkey::text, ' ')::int2[]) WITH ORDINALITY as u(elem, ordinality)
                            WHERE pa.attnum = u.elem
                        ) as idx
                        WHERE pns.nspname = '%s'
                        AND pct.relname = '%s'
                        ORDER BY pci.relname, pct.relname, idx.index;""" % (schema_name, table_name)
        cursor.execute(column_query)
        index_columns = {oid: [] for oid in index_names}
        for oid, _, column_name in cursor.fetchall():
            index_columns[oid].append(column_dict[column_name])
        # Compute prefix distinct counts per index.
        # Single-column prefixes use the column's own distinct_count; only multi-column prefixes need a query.
        # Each index gets its own query so PostgreSQL can use that index for the scan.
        prefix_distinct_counts = {}
        for oid in index_names:
            cols = index_columns[oid]
            unique = index_unique[oid]
            prefix_distinct_counts[oid] = [cols[0].distinct_count()] + [None] * (len(cols) - 1)
            if unique:
                prefix_distinct_counts[oid][-1] = cardinality
            if len(cols) > 1:
                # Query only the prefixes we don't already know.
                count_exprs = []
                expr_indices = []
                for i in range(2, len(cols) + 1):
                    if prefix_distinct_counts[oid][i - 1] is not None:
                        continue
                    col_names = ", ".join(c.name() for c in cols[:i])
                    count_exprs.append("COUNT(DISTINCT (%s))" % col_names)
                    expr_indices.append(i - 1)
                if count_exprs:
                    prefix_query = "SELECT %s FROM %s;" % (", ".join(count_exprs), table_name)
                    cursor.execute(prefix_query)
                    results = cursor.fetchone()
                    for i, result in enumerate(results):
                        prefix_distinct_counts[oid][expr_indices[i]] = result

        indexes = []
        primary_key = None
        for oid, index_name in index_names.items():
            level_query = "SELECT (bt_metap('%s')).level;" % index_name
            cursor.execute(level_query)
            level = cursor.fetchone()[0]
            index = Index(index_name, index_columns[oid], index_unique[oid], level, index_pages[oid], index_tuples[oid],
                          prefix_distinct_counts[oid])
            if oid == primary_key_id:
                primary_key = index
            else:
                indexes.append(index)
        cursor.close()
        return primary_key, indexes




