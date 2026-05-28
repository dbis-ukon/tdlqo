from typing import Optional

from schemas.data_types.data_type import DataType


class StringDataType(DataType):
    def __init__(self, name: str, max_length: Optional[int] = None):
        super().__init__(name)
        self.max_length = max_length

    def wrap_sql(self, value: str):
        pass

    def width(self) -> int:
        # TODO: Frankly, these are horrible simplifications of the actual storage requirements including TOAST, compression, etc.
        # Right now these guesses are not used, instead we simply ask PostgreSQL for the actual width of string columns, when initializing a Column.
        if self.max_length is None:
            return 16
        else:
            if self.max_length < 126:
                return self.max_length + 1
            else:
                return self.max_length + 4
