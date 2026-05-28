from decimal import Decimal

from queries.predicates.literals.literal import Literal


class NumberLiteral(Literal):
    def __init__(self, value: Decimal):
        super().__init__(str(value))
        self._value = value

    def value(self) -> Decimal:
        return self._value
