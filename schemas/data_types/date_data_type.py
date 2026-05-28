import datetime

from schemas.data_types.data_type import DataType


class DateDataType(DataType):
    def __init__(self):
        super().__init__("date")

    def wrap_sql(self, value: datetime.date):
        pass

    def width(self) -> int:
        return 4
