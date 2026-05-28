from typing import List, Tuple

from queries.predicates.literals.literal import Literal


class TupleLiteral(Literal):
    def __init__(self, values: List[Literal]):
        super().__init__(self.wrap_string(values))
        self._values = values

    @staticmethod
    def wrap_string(values: List[Literal]) -> str:
        return "(" + ", ".join([value.alias_string() for value in values]) + ")"

    def value(self) -> Tuple[Literal]:
        return tuple(value.value() for value in self._values)

