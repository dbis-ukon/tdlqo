from abc import abstractmethod


class DataType:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name

    @abstractmethod
    def wrap_sql(self, value):
        pass

    @abstractmethod
    def width(self) -> int:
        pass

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not isinstance(other, DataType):
            return False
        return self._name == other._name
