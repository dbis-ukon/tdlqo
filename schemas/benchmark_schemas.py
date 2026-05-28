from typing import Optional

from schemas.schema import Schema


def imdb_schema(port: Optional[int] = None) -> Schema:
    return Schema.build_from_connection("imdb_ceb", fk_name="imdb_schema", port=port)


def stats_ceb_schema(port: Optional[int] = None) -> Schema:
    return Schema.build_from_connection("stats-ceb", port=port)


def stack_schema(port: Optional[int] = None) -> Schema:
    return Schema.build_from_connection("stack", port=port)



