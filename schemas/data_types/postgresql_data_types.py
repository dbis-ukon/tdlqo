import math
from typing import Dict, Optional

from schemas.data_types.data_type import DataType
from schemas.data_types.date_data_type import DateDataType
from schemas.data_types.numeric_data_type import NumericDataType
from schemas.data_types.string_data_type import StringDataType
from schemas.data_types.timestamp_data_type import TimestampDataType


def get_postgresql_data_type(name: str, numeric_precision: Optional[int] = None, maximum_length: Optional[int] = None) -> DataType:
    if name == "integer":
        return NumericDataType("integer", 4)
    elif name == "smallint":
        return NumericDataType("smallint", 2)
    elif name == "numeric":
        assert numeric_precision is not None
        ndigits = math.ceil(numeric_precision / 4)
        width = 4 + 2 * ndigits
        return NumericDataType("numeric", width)
    elif name == "double precision" or name == "double":
        return NumericDataType("double precision", 8)
    elif name == "character varying":
        return StringDataType("character varying", maximum_length)
    elif name == "character":
        return StringDataType("character", maximum_length)
    elif name == "date":
        return DateDataType()
    elif name == "timestamp" or name == "timestamp without time zone" or name == "timestamp with time zone":
        return TimestampDataType()
    else:
        raise ValueError(f"Unsupported data type: {name}")
