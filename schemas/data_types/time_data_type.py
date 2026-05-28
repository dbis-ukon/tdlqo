import datetime

from schemas.data_types.data_type import DataType


class TimeDataType(DataType):
    def __init__(self):
        super().__init__("time")

    def wrap_sql(self, value: datetime.time):
        pass

    def width(self) -> int:
        return 8
