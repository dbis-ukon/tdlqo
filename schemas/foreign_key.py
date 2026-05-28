from typing import Dict, Tuple

from schemas.column import Column
from schemas.table import Table


class ForeignKey:
    def __init__(self, name: str, foreign_key_table: Table, primary_key_table: Table, mapping: Dict[Column, Column]):
        self._name = name
        self._foreign_key_table = foreign_key_table
        self._primary_key_table = primary_key_table
        self._mapping = mapping

    def name(self) -> str:
        return self._name

    def foreign_key_table(self) -> Table:
        return self._foreign_key_table

    def primary_key_table(self) -> Table:
        return self._primary_key_table

    def mapping(self) -> Dict[Column, Column]:
        return self._mapping
