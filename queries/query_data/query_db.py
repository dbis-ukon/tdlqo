import datetime
from typing import Optional, Any, Dict, List

import psycopg2


class QueryDB:
    def __init__(self, name: str, port: int):
        self._connection = psycopg2.connect(host="localhost", database=name, user="silvan", port=port)

    def commit(self):
        self._connection.commit()

    @staticmethod
    def _wrap_value(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, str):
            return "'%s'" % value.replace("'", "''")
        if isinstance(value, datetime.datetime):
            return "'%s'" % value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, datetime.date):
            return "'%s'" % value.strftime("%Y-%m-%d")
        if isinstance(value, datetime.time):
            return "'%s'" % value.strftime("%H:%M:%S")
        return str(value)

    @staticmethod
    def _build_insert(table: str, values: Dict[str, Any], return_id: bool) -> str:
        column_string = ", ".join(values)
        value_string = ", ".join([QueryDB._wrap_value(values[c]) for c in values])
        options = ""
        if return_id:
            options += " RETURNING id"
        insert_query = "INSERT INTO %s(%s) VALUES (%s)%s;" % (table, column_string, value_string, options)
        return insert_query

    def insert(self, table: str, values: Dict[str, Any], return_id: bool = False) -> Optional[int]:
        insert_query = QueryDB._build_insert(table, values, return_id)
        cursor = self._connection.cursor()
        cursor.execute(insert_query)
        if return_id:
            result = cursor.fetchone()
            cursor.close()
            return result[0]
        cursor.close()
        return None

    def search(self, search_query: str) -> List[tuple]:
        cursor = self._connection.cursor()
        cursor.execute(search_query)
        result = cursor.fetchall()
        cursor.close()
        return result


