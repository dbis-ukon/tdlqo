import datetime

from schemas.data_types.data_type import DataType


class TimestampDataType(DataType):
    def __init__(self):
        super().__init__("timestamp")

    def wrap_sql(self, value: datetime.datetime):
        pass

    def width(self) -> int:
        return 8
