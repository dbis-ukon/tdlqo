

from __future__ import annotations

from typing import Optional

import psycopg2


unit_multipliers = {
    'PB': 1024**5,
    'TB': 1024**4,
    'GB': 1024**3,
    'MB': 1024**2,
    'KB': 1024,
    'B': 1
}


class PostgreSQLConfiguration:
    def __init__(self,
                 cpu_tuple_cost: float,
                 cpu_index_tuple_cost: float,
                 cpu_operator_cost: float,
                 seq_page_cost: float,
                 random_page_cost: float,
                 work_mem: Optional[int],
                 hash_mem_multiplier: float,
                 block_size: int,
                 effective_cache_size: int
                 ):
        self.cpu_tuple_cost = cpu_tuple_cost
        self.cpu_index_tuple_cost = cpu_index_tuple_cost
        self.cpu_operator_cost = cpu_operator_cost
        self.seq_page_cost = seq_page_cost
        self.random_page_cost = random_page_cost
        self.work_mem = work_mem
        self.hash_mem_multiplier = hash_mem_multiplier
        self.block_size = block_size
        self.effective_cache_size = effective_cache_size

    @staticmethod
    def default_configuration() -> PostgreSQLConfiguration:
        return PostgreSQLConfiguration(
            cpu_tuple_cost=0.01,
            cpu_index_tuple_cost=0.005,
            cpu_operator_cost=0.0025,
            seq_page_cost=1,
            random_page_cost=4,
            work_mem=4 * 2 ** 20,  # 4 MB
            hash_mem_multiplier=2.0,
            block_size=8192,
            effective_cache_size=524288
        )

    @staticmethod
    def from_connection(connection: psycopg2._psycopg.connection) -> PostgreSQLConfiguration:
        cursor = connection.cursor()
        cursor.execute("SHOW cpu_tuple_cost;")
        cpu_tuple_cost = float(cursor.fetchone()[0])
        cursor.execute("SHOW cpu_index_tuple_cost;")
        cpu_index_tuple_cost = float(cursor.fetchone()[0])
        cursor.execute("SHOW cpu_operator_cost;")
        cpu_operator_cost = float(cursor.fetchone()[0])
        cursor.execute("SHOW seq_page_cost;")
        seq_page_cost = float(cursor.fetchone()[0])
        cursor.execute("SHOW random_page_cost;")
        random_page_cost = float(cursor.fetchone()[0])
        cursor.execute("SHOW work_mem;")
        work_mem_string = cursor.fetchone()[0].split()[0]
        work_mem = parse_size_string(work_mem_string)
        cursor.execute("SHOW hash_mem_multiplier;")
        hash_mem_multiplier = float(cursor.fetchone()[0])
        cursor.execute("SHOW block_size;")
        block_size = int(cursor.fetchone()[0])
        cursor.execute("SHOW effective_cache_size;")
        effective_cache_size_string = cursor.fetchone()[0].split()[0]
        effective_cache_size = parse_size_string(effective_cache_size_string)
        cursor.close()
        return PostgreSQLConfiguration(
            cpu_tuple_cost=cpu_tuple_cost,
            cpu_index_tuple_cost=cpu_index_tuple_cost,
            cpu_operator_cost=cpu_operator_cost,
            seq_page_cost=seq_page_cost,
            random_page_cost=random_page_cost,
            work_mem=work_mem,
            hash_mem_multiplier=hash_mem_multiplier,
            block_size=block_size,
            effective_cache_size=effective_cache_size
        )

    def copy(self) -> PostgreSQLConfiguration:
        return PostgreSQLConfiguration(
            cpu_tuple_cost=self.cpu_tuple_cost,
            cpu_index_tuple_cost=self.cpu_index_tuple_cost,
            cpu_operator_cost=self.cpu_operator_cost,
            seq_page_cost=self.seq_page_cost,
            random_page_cost=self.random_page_cost,
            work_mem=self.work_mem,
            hash_mem_multiplier=self.hash_mem_multiplier,
            block_size=self.block_size,
            effective_cache_size=self.effective_cache_size
        )


def parse_size_string(size_str: str) -> int:
    size_str = size_str.strip().upper()
    for unit in unit_multipliers:
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)].strip())
            return number * unit_multipliers[unit]
    raise ValueError(f"Invalid size string: {size_str}")


