from schemas.data_types.data_type import DataType


class NumericDataType(DataType):
    def __init__(self, name: str, width: int):
        super().__init__(name)
        self._width = width

    def wrap_sql(self, value):
        return str(value)

    def width(self) -> int:
        return self._width
