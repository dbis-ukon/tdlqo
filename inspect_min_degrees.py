"""Print single-column min_degree for every FK column, alongside max_degree
and the distinct-count comparison against the referenced column. The goal is
to eyeball how often min_degree > 0 — that's where a MIN-mode baseline using
min_degree would actually gain signal over the current hard-coded 0.

Rule: under referential integrity, distinct(fk_col) <= distinct(pk_col). When
strictly less, some referenced values are unreferenced, so the effective
min_degree is 0 (no SQL needed). When equal, every referenced value is
covered and the in-table MIN(COUNT(*)) GROUP BY fk_col gives the real bound.
"""

from schemas.benchmark_schemas import stack_schema, imdb_schema


schema = stack_schema(port=5443)

# Per (table, column), track the largest pk_distinct it must cover and any
# missing-stat flag, so we evaluate each FK column once.
fk_columns = {}
for fk in schema.foreign_keys():
    child_table = fk.foreign_key_table()
    for fk_col, pk_col in fk.mapping().items():
        key = (child_table, fk_col)
        entry = fk_columns.get(key)
        if entry is None:
            entry = {"max_pk_distinct": None, "missing_stat": False}
            fk_columns[key] = entry
        pk_distinct = pk_col.distinct_count()
        if pk_distinct is None or fk_col.distinct_count() is None:
            entry["missing_stat"] = True
            continue
        if entry["max_pk_distinct"] is None or pk_distinct > entry["max_pk_distinct"]:
            entry["max_pk_distinct"] = pk_distinct

rows = []
for (child_table, fk_col), entry in fk_columns.items():
    fk_distinct = fk_col.distinct_count()
    if entry["missing_stat"]:
        min_degree = None
        note = "missing distinct stat"
    elif fk_distinct < entry["max_pk_distinct"]:
        min_degree = 0
        note = "uncovered referenced values"
    else:
        min_degree = fk_col.min_degree()
        note = "" if min_degree is not None else "no stored min_degree"
    rows.append((
        child_table.name(),
        fk_col.name(),
        fk_distinct,
        fk_col.max_degree(),
        min_degree,
        note,
    ))

header = "%-22s %12s %12s %12s   %s" % ("table.column", "distinct", "max_degree", "min_degree", "note")
print(header)
print("-" * len(header))
for table_name, column_name, fk_d, max_d, min_d, note in sorted(rows):
    label = "%s.%s" % (table_name, column_name)
    min_d_str = "-" if min_d is None else str(min_d)
    max_d_str = "-" if max_d is None else str(max_d)
    fk_d_str = "-" if fk_d is None else str(fk_d)
    print("%-22s %12s %12s %12s   %s" % (label, fk_d_str, max_d_str, min_d_str, note))

print()
nonzero = [r for r in rows if r[4] is not None and r[4] > 0]
print("FK columns with min_degree > 0 (potential signal gain): %d / %d" % (len(nonzero), len(rows)))
for table_name, column_name, _, _, min_d, _ in sorted(nonzero):
    print("  %s.%s  min_degree=%d" % (table_name, column_name, min_d))
