
from queries.predicates.literals.literal import Literal


class NullLiteral(Literal):
    def __init__(self):
        super().__init__("NULL")

    def value(self) -> None:
        return None

