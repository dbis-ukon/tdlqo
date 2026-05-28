
from __future__ import annotations

from typing import Optional, Dict, Tuple

from queries.predicates.join import Join
from queries.table_occurrence import TableOccurrence
from schemas.index import Index


class Requirements:
    def __init__(self,
                 force_index_scan: Optional[Tuple[Index, bool]] = None,
                 sorted: Optional[Join] = None):
        self.force_index_scan = force_index_scan
        self.sorted = sorted

    def has_no_requirement(self) -> bool:
        return self.force_index_scan is None and self.sorted is None

    def possibly_equal(self, other: Requirements) -> Optional[bool]:
        if self.force_index_scan != other.force_index_scan:
            return False
        if self.sorted is None and other.sorted is None:
            return True
        elif self.sorted is not None and other.sorted is not None:
            own_signature = self.sorted.signature()
            other_signature = other.sorted.signature()
            if own_signature == other_signature:
                return None
            else:
                return False
        else:
            return False

    def assignment_equal(self, other: Requirements, table_occurrence_mapping: Dict[TableOccurrence, TableOccurrence]) -> bool:
        if self.force_index_scan != other.force_index_scan:
            return False
        if self.sorted is None and other.sorted is None:
            return True
        elif self.sorted is not None and other.sorted is not None:
            return self.sorted.syntactically_equal(other.sorted, table_occurrence_mapping)
        else:
            return False

    def __hash__(self):
        if self.force_index_scan is None:
            index_hash = 0
        else:
            index_hash = hash(self.force_index_scan)
        if self.sorted is None:
            sorted_hash = 0
        else:
            sorted_hash = self.sorted.simple_hash()
        return index_hash ^ sorted_hash





