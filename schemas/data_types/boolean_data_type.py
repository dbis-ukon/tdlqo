
from schemas.data_types.data_type import DataType


class BooleanType(DataType):
    def __init__(self):
        super().__init__("boolean")

    def wrap_sql(self, value: bool):
        pass

    def width(self) -> int:
        return 1
