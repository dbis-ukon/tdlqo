from queries.predicates.literals.literal import Literal


class StringLiteral(Literal):
    def __init__(self, value: str):
        super().__init__(self.wrap_string(value))
        self._value = value

    @staticmethod
    def wrap_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def value(self) -> str:
        return self._value
