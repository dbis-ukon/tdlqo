from abc import abstractmethod
from typing import Any

from queries.predicates.expression import Expression


class Literal(Expression):
    def __init__(self, pattern: str):
        super().__init__(pattern, [])

    @abstractmethod
    def value(self) -> Any:
        pass

