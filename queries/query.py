from abc import abstractmethod
from typing import Optional


class Query:
    def __init__(self, query_text: Optional[str]):
        self._query_text = query_text

    def query_text(self) -> str:
        if self._query_text is None:
            self._query_text = self._get_query_text()
        return self._query_text

    @abstractmethod
    def _get_query_text(self) -> str:
        raise NotImplementedError()
