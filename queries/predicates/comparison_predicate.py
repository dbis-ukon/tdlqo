
from __future__ import annotations

from decimal import Decimal
from typing import FrozenSet, Optional, Dict

from queries.predicates.comparison_operator import ComparisonOperator, COMPARISON_OPERATOR_IS
from queries.predicates.predicate import Predicate
from queries.table_occurrence import TableOccurrence
from schemas.column import Column
from sample_encoders.table_row import TableRow
from schemas.data_types.data_type import DataType
from schemas.data_types.numeric_data_type import NumericDataType
from schemas.data_types.string_data_type import StringDataType


class ComparisonPredicate(Predicate):
    """Comparison between two columns of the same table occurrence, e.g. cn.name_pcode_nf = cn.name_pcode_sf."""

    def __init__(self, table_occurrence: TableOccurrence, left_column: Column, operator: ComparisonOperator, right_column: Column,
                 left_column_cast_data_type: Optional[DataType] = None, right_column_cast_data_type: Optional[DataType] = None):
        left_string = _column_pattern(left_column, left_column_cast_data_type)
        right_string = _column_pattern(right_column, right_column_cast_data_type)
        super().__init__(left_string + " " + operator.symbol() + " " + right_string, [table_occurrence, table_occurrence])
        self._table_occurrence = table_occurrence
        self._left_column = left_column
        self._operator = operator
        self._right_column = right_column
        self._left_column_cast_data_type = left_column_cast_data_type
        self._right_column_cast_data_type = right_column_cast_data_type

    def table_occurrence(self) -> TableOccurrence:
        return self._table_occurrence

    def left_column(self) -> Column:
        return self._left_column

    def right_column(self) -> Column:
        return self._right_column

    def operator(self) -> ComparisonOperator:
        return self._operator

    def left_column_cast_type(self) -> Optional[DataType]:
        return self._left_column_cast_data_type

    def right_column_cast_type(self) -> Optional[DataType]:
        return self._right_column_cast_data_type

    def canonical_operands(self):
        """Operands of a symmetric operator are unordered, so that a = b and b = a compare and hash equal."""
        left = (self._left_column, self._left_column_cast_data_type)
        right = (self._right_column, self._right_column_cast_data_type)
        if self._operator.anti_symmetric_operator() is self._operator:
            return frozenset({left, right})
        return left, right

    def _get_simple_hash(self) -> int:
        return hash((self._operator, self.canonical_operands()))

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        if not isinstance(other, ComparisonPredicate):
            return False
        if self._operator != other.operator() or self.canonical_operands() != other.canonical_operands():
            return False
        table_occurrence = self._table_occurrence
        other_table_occurrence = other.table_occurrence()
        if table_occurrence_mapping is None:
            return table_occurrence == other_table_occurrence
        return table_occurrence == table_occurrence_mapping[other_table_occurrence]

    def _check_normal_form(self) -> bool:
        return True

    def _get_normal_form(self) -> Optional[Predicate]:
        return self

    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        left_value = _cast_value(table_row.values[self._left_column], self._left_column_cast_data_type)
        right_value = _cast_value(table_row.values[self._right_column], self._right_column_cast_data_type)
        if right_value is None and self._operator != COMPARISON_OPERATOR_IS:
            return None
        return self._operator.evaluate(left_value, right_value)

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        return ComparisonPredicate(replacements.get(self._table_occurrence, self._table_occurrence), self._left_column, self._operator,
                                   self._right_column, self._left_column_cast_data_type, self._right_column_cast_data_type)

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        if self._table_occurrence == table_occurrence:
            return frozenset({self._left_column, self._right_column})
        return frozenset()


def _column_pattern(column: Column, cast_data_type: Optional[DataType]) -> str:
    column_string = "%s." + column.name()
    if cast_data_type is not None:
        column_string = "(" + column_string + ")::" + cast_data_type.name()
    return column_string


def _cast_value(value, cast_data_type: Optional[DataType]):
    if cast_data_type is None or value is None:
        return value
    if isinstance(cast_data_type, NumericDataType):
        return Decimal(value)
    elif isinstance(cast_data_type, StringDataType):
        return str(value)
    else:
        raise NotImplementedError(f"Unsupported cast to data type: {cast_data_type.name()}")
