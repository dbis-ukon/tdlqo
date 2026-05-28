
from __future__ import annotations
from typing import Dict, Any, List

from schemas.column import Column


class TableRow:
    def __init__(self, values: Dict[Column, Any]):
        self.values = values

    @staticmethod
    def build_table_row(columns: List[Column], postgresql_tuple: tuple) -> TableRow:
        values = {}
        for column, value in zip(columns, postgresql_tuple):
            values[column] = value
        return TableRow(values)









