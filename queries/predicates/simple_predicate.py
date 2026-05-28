
from __future__ import annotations

from decimal import Decimal
from typing import FrozenSet, Optional, Dict, Type

from queries.predicates.comparison_operator import ComparisonOperator, COMPARISON_OPERATOR_IN
from queries.predicates.literals.literal import Literal
from queries.predicates.predicate import Predicate
from queries.table_occurrence import TableOccurrence
from schemas.column import Column
from sample_encoders.table_row import TableRow
from schemas.data_types.data_type import DataType
from schemas.data_types.numeric_data_type import NumericDataType
from schemas.data_types.string_data_type import StringDataType


class SimplePredicate(Predicate):
    def __init__(self, table_occurrence: TableOccurrence, column: Column, operator: ComparisonOperator, value: Literal, column_cast_data_type: Optional[DataType] = None):
        column_string = "%s." + column.name()
        if column_cast_data_type is not None:
            column_string = "(" + column_string + ")::" + column_cast_data_type.name()
        super().__init__(column_string + " " + operator.symbol() + " " + value.pattern(), [table_occurrence])
        self._table_occurrence = table_occurrence
        self._column = column
        self._operator = operator
        self._value = value
        self._column_cast_data_type = column_cast_data_type

    def table_occurrence(self) -> TableOccurrence:
        return self._table_occurrence

    def column(self) -> Column:
        return self._column

    def operator(self) -> ComparisonOperator:
        return self._operator

    def value(self) -> Literal:
        return self._value

    def column_cast_type(self) -> Optional[DataType]:
        return self._column_cast_data_type

    @staticmethod
    def from_atomic_predicate(predicate: Predicate) -> Optional[SimplePredicate]:
        pass

    def canonical_value(self):
        if self._operator == COMPARISON_OPERATOR_IN:
            return frozenset(self._value.value())
        else:
            return self._value.value()

    def _get_simple_hash(self) -> int:
        return hash((self._column, self._operator, self.canonical_value(), self._column_cast_data_type))

    def syntactically_equal(self, other: Predicate, table_occurrence_mapping: Optional[Dict[TableOccurrence, TableOccurrence]] = None) -> bool:
        if not isinstance(other, SimplePredicate):
            return False
        if self._column != other.column() or self._operator != other.operator() or self.canonical_value() != other.canonical_value():
            return False
        if self._column_cast_data_type != other.column_cast_type():
            return False
        table_occurrence = self._table_occurrence
        other_table_occurrence = other.table_occurrence()
        if table_occurrence_mapping is None:
            return table_occurrence == other_table_occurrence
        return table_occurrence == table_occurrence_mapping[other_table_occurrence]

    # TODO: Potentially, eliminate derived comparison operators like "!=" and ">"
    def _check_normal_form(self) -> bool:
        return True

    def _get_normal_form(self) -> Optional[Predicate]:
        return self

    def evaluate(self, table_row: TableRow) -> Optional[bool]:
        left_value = table_row.values[self._column]
        if self._column_cast_data_type is not None and left_value is not None:
            if isinstance(self._column_cast_data_type, NumericDataType):
                left_value = Decimal(left_value)
            elif isinstance(self._column_cast_data_type, StringDataType):
                left_value = str(left_value)
            else:
                raise NotImplementedError(f"Unsupported cast to data type: {self._column_cast_data_type.name()}")
        return self._operator.evaluate(left_value, self._value.value())

    def replace(self, replacements: Dict[TableOccurrence, TableOccurrence]) -> Predicate:
        return SimplePredicate(replacements.get(self._table_occurrence, self._table_occurrence), self._column, self._operator, self._value, self._column_cast_data_type)

    def columns_for(self, table_occurrence: TableOccurrence) -> FrozenSet[Column]:
        if self._table_occurrence == table_occurrence:
            return frozenset({self._column})
        return frozenset()

