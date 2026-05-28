from typing import FrozenSet, List, Optional, Callable

import psycopg2

from schemas.postgresql_configuration import PostgreSQLConfiguration
from schemas.table import Table

from schemas.foreign_key import ForeignKey


class Schema:
    def __init__(self,
                 postgresql_configuration: PostgreSQLConfiguration,
                 database_name: str,
                 schema_name: str,
                 connection_function: Callable[[], psycopg2._psycopg.connection],
                 tables: List[Table],
                 foreign_keys: List[ForeignKey]) -> None:
        self._postgresql_configuration = postgresql_configuration
        self._database_name = database_name
        self._schema_name = schema_name
        self._connection_function = connection_function
        self._tables = frozenset(tables)
        self._table_dict = {table.name(): table for table in tables}
        self._foreign_keys = frozenset(foreign_keys)
        self._foreign_keys_from = {table: [] for table in tables}
        self._foreign_keys_to = {table: [] for table in tables}
        for foreign_key in foreign_keys:
            self._foreign_keys_from[foreign_key.foreign_key_table()].append(foreign_key)
            self._foreign_keys_to[foreign_key.primary_key_table()].append(foreign_key)

    def connection(self):
        return self._connection_function()

    def database_name(self) -> str:
        return self._database_name

    def schema_name(self) -> str:
        return self._schema_name

    def postgresql_configuration(self) -> PostgreSQLConfiguration:
        return self._postgresql_configuration

    def tables(self) -> FrozenSet[Table]:
        return self._tables

    def table(self, name: str) -> Table:
        assert(name in self._table_dict)
        return self._table_dict[name]

    def foreign_keys(self) -> FrozenSet[ForeignKey]:
        return self._foreign_keys

    def foreign_keys_from(self, table: Table) -> List[ForeignKey]:
        return self._foreign_keys_from[table]

    def foreign_keys_to(self, table: Table) -> List[ForeignKey]:
        return self._foreign_keys_to[table]

    @staticmethod
    def build_from_connection(name: str, fk_name: Optional = None, schema: str = "public", port: int = 5432):
        connection_function = lambda: psycopg2.connect(host="localhost", database=name, user="silvan", password="postgres", port=port)
        connection = connection_function()
        if fk_name is None:
            fk_connection = connection
        else:
            fk_connection = psycopg2.connect(host="localhost", database=fk_name, user="silvan", password="postgres", port=port)
        postgresql_configuration = PostgreSQLConfiguration.from_connection(connection)
        cursor = connection.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pageinspect;")
        cursor.close()
        tables = Schema._get_tables(connection, schema)
        foreign_keys = Schema._get_foreign_keys(fk_connection, schema, tables)

        return Schema(postgresql_configuration, name, schema, connection_function, tables, foreign_keys)

    @staticmethod
    def _get_tables(connection, schema: str) -> List[Table]:
        tables = []

        cursor = connection.cursor()
        table_query = """SELECT table_name
                         FROM information_schema.tables
                         WHERE table_schema = '%s' AND table_name != 'plan_cache'
                         ORDER BY table_name;""" % schema
        cursor.execute(table_query)
        table_names = [result[0] for result in cursor.fetchall()]
        for i, table_name in enumerate(table_names):
            print(f"Building table {table_name} ({i + 1}/{len(table_names)})")
            tables.append(Table.build_from_connection(connection, table_name, schema))
        cursor.close()

        return tables

    @staticmethod
    def _get_foreign_keys(connection, schema: str, tables: List[Table]) -> List[ForeignKey]:
        table_dict = {table.name(): table for table in tables}
        foreign_keys = []

        cursor = connection.cursor()

        # Improved query using referential_constraints to guarantee correct column mapping
        foreign_key_query = """
            SELECT 
                tc.constraint_name, 
                kcu.table_name AS foreign_table, 
                kcu.column_name AS foreign_column, 
                kcu_target.table_name AS primary_table, 
                kcu_target.column_name AS primary_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu 
                ON tc.constraint_name = kcu.constraint_name 
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.referential_constraints AS rc 
                ON tc.constraint_name = rc.constraint_name 
                AND tc.constraint_schema = rc.constraint_schema
            JOIN information_schema.key_column_usage AS kcu_target 
                ON rc.unique_constraint_name = kcu_target.constraint_name 
                AND rc.unique_constraint_schema = kcu_target.constraint_schema 
                AND kcu.position_in_unique_constraint = kcu_target.ordinal_position
            WHERE tc.constraint_schema = '%s'
            AND tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.constraint_name;
        """ % schema

        cursor.execute(foreign_key_query)
        fk_references = {}

        for constraint_name, ft, fc, pt, pc in cursor.fetchall():
            if ft not in table_dict or pt not in table_dict:
                continue

            foreign_key_table = table_dict[ft]
            foreign_key_column = foreign_key_table.column(fc)
            primary_key_table = table_dict[pt]
            primary_key_column = primary_key_table.column(pc)

            if constraint_name not in fk_references:
                fk_references[constraint_name] = (foreign_key_table, primary_key_table, {})

            fk_references[constraint_name][2][foreign_key_column] = primary_key_column

        for constraint_name, (foreign_key_table, primary_key_table, mapping) in fk_references.items():
            foreign_key = ForeignKey(constraint_name, foreign_key_table, primary_key_table, mapping)
            foreign_keys.append(foreign_key)

        cursor.close()
        return foreign_keys
