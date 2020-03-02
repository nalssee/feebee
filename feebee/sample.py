
import sqlite3
import sqlparse

from moz_sql_parser import parse

sql0 = """
select a
from c x, d
where a >= 10
order by t
"""
sql1 = """
select a.col1, B.col2
from table1 as a
left join table2 as b
on a.col2 > b.col2
left join table3 c
on a.col20 = c.col3 and a.col1 > b.col1 + c.col4
where table1.col1 = 1992
order by c.col
"""


sql2 = """
select  K.a,K.b as xxx from (select H.b from (select G.c from (select F.d from
(select E.e from ab as A, B, C, D, E), F), G), H), I, J, K order by 1,2;
"""


sql3 = """
select a.col1
from table a
left join table2 as b using(col1)
"""


# conn = sqlite3.connect('sample.db')
# c = conn.cursor()

# # Create table
# c.execute('''CREATE TABLE if not exists stocks
#              (date text, trans text, symbol text, qty real, price real)''')

# # Insert a row of data
# c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
# c.execute("create index if not exists stock_idx on stocks(date)")


# # Save (commit) the changes
# conn.commit()

# for row in c.execute("select * from stocks"):
#     print(row)



# # We can also close the connection if we are done with it.
# # Just be sure any changes have been committed or they will be lost.
# conn.close()


# for item in parsed.tokens:
#     idx = parsed.token_index(item)
#     (_, prev) = parsed.token_prev(idx)
#     print(item.__repr__(),' | ',  prev_keyword(item, parsed).__repr__())

# def _is_subselect(parsed):
#     if not parsed.is_group:
#         return False
#     for item in parsed.tokens:
#         if item.ttype is sqlparse.tokens.DML and item.value.upper() == 'SELECT':
#             return True
#     return False


# def _get_prev_keyword(item, parsed):
#     idx = parsed.token_index(item)
#     found = None
#     while idx and not found:
#         idx -= 1
#         if parsed.tokens[idx].ttype == sqlparse.tokens.Keyword:
#             found = parsed.tokens[idx]
#     return found


# def _get_tables(parsed):
#     def _gen(parsed):
#         for item in parsed.tokens:
#             prev = _get_prev_keyword(item, parsed)
#             if prev and ("JOIN" in prev.value.upper() or "FROM" in prev.value.upper()):
#                 if isinstance(item, sqlparse.sql.Identifier):
#                     yield item.get_real_name()
#                 elif isinstance(item, sqlparse.sql.IdentifierList):
#                     for x in item:
#                         if isinstance(x, sqlparse.sql.Identifier) or isinstance(x, sqlparse.sql.IdentifierList):
#                             yield x.get_real_name()
#                         elif _is_subselect(x):
#                             yield from _gen(x)

#                 elif _is_subselect(item):
#                     yield from _gen(item)
#     return list(dict.fromkeys(_gen(parsed)))


# def _get_cols_for_matching(parsed):
#     def _gen(parsed):
#         for item in parsed.tokens:
#             prev = _get_prev_keyword(item, parsed)
#             if prev and "ON" in prev.value.upper():
#                 if isinstance(item, sqlparse.sql.Comparison):
#                     for x in item:
#                         if isinstance(x, sqlparse.sql.Identifier):
#                             print(x)
#                             yield x
#                 #     yield item
#                 # elif isinstance(item, sqlparse.sql.IdentifierList):
#                 #     for x in item:
#                 #         if isinstance(x, sqlparse.sql.Identifier) or isinstance(x, sqlparse.sql.IdentifierList):
#                 #             yield x

#     return list(_gen(parsed))


# for x in _get_tables(sqlparse.parse(sql2)[0]):
# _get_cols_for_matching(sqlparse.parse(sql1)[0])
#     print(x)


#  ambiguous get
def _dget(d, sym):
    for k, v in d.items():
        if sym in k:
            return (k, v)
    return (None, None)


# def _filter_clauses(xs, symbol):
#     return [x for x in xs if _contains(x, symbol)]


def _get_tables_from_sql(parsed):
    tables = parsed['from']
    if isinstance(tables, str):
        yield tables
    # list of tables
    else:
        for table in tables:
            if isinstance(table, str):
                yield table
            else:
                k, v = _dget(table, 'value')
                if k:
                    if isinstance(v, str):
                        yield v
                    # subselect
                    else:
                        yield from _get_tables_from_sql(v)

                else:
                    k, v = _dget(table, 'join')
                    if k:
                        k1, v1 = _dget(v, 'value')
                        if k1:
                            if isinstance(v1, str):
                                yield v1
                            # subselect
                            else:
                                yield from _get_tables_from_sql(v1)


def _get_matching_cols(parsed):
    tables = parsed['from']


# print(parse(sql1))
print(parse(sql0))
print(list(_get_tables_from_sql(parse(sql1))))

# parsed = parse(sql2)


# for k, v in parsed.items():
#     print(k, v)