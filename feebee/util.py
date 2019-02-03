import os
import locale
import csv
from openpyxl import load_workbook
import statistics as st
from scipy.stats import ttest_1samp
from datetime import timedelta
import numpy as np
from itertools import zip_longest, accumulate, chain, groupby
import pandas as pd
import ciso8601
import io
import statsmodels.api as sm


ORDERS = """\
orderid,customerid,employeeid,orderdate,shipperid
10248,90,5,1996-07-04,3
10252,76,4,1996-07-09,2
10249,81,6,1996-07-05,1
10253,34,3,1996-07-10,2
10250,34,4,1996-07-08,2
10251,84,3,1996-07-08,1
10254,14,5,1996-07-11,2\
"""


def lag(cols, datecol, ns, add1fn=None, max_missings=10_000):
    """ create columns for lags and leads

    >>> rs = _str2dicts(ORDERS)
    >>> result = lag('orderid, customerid', 'orderdate', [1, 2, -1])(rs)
    >>> len(rs) == len(result)
    True
    >>> list(result[0].keys())
    ['orderid', 'customerid', 'employeeid', 'orderdate', 'shipperid', 'orderid_1', 'customerid_1', 'orderid_2', 'customerid_2', 'orderid_1n', 'customerid_1n']

    >>> xs = [r['orderid'] for r in result]
    >>> xs
    ['10248', '10249', '10250', '10251', '10252', '10253', '10254']
    >>> ys = [r['orderid_2'] for r in result]
    >>> ys[2:] == xs[:5]
    True
    >>> ys[:2]
    ['', '']
    >>> zs = [r['orderid_1n'] for r in result]
    >>> xs[1:] == zs[:6]
    True
    >>> zs[6:]
    ['']
    >>> # Pass add1fn
    >>> del rs[2] # raise exception for duplicates, so.
    >>> del rs[-2] # Just for the heck of it
    >>> result = lag('orderid, customerid', 'orderdate', [1, 2, -1], ndate(1))(rs)
    >>> [r['orderdate'] for r in result]
    ['1996-07-04', '1996-07-05', '1996-07-06', '1996-07-07', '1996-07-08', '1996-07-09', '1996-07-10', '1996-07-11']
    >>> [r['orderid'] for r in result]
    ['10248', '10249', '', '', '10251', '10252', '', '10254']
    >>> [r['orderid_2'] for r in result]
    ['', '', '10248', '10249', '', '', '10251', '10252']
    >>> [r['orderid_1n'] for r in result]
    ['10249', '', '', '10251', '10252', '', '10254', '']
    """
    cols = listify(cols)
    ns = listify(ns)

    def fn(rs):
        suffix = "_"
        rs.sort(key=lambda r: r[datecol])
        if add1fn:
            rs1 = [rs[0]]
            for r1, r2 in zip(rs, rs[1:]):
                d1, d2 = r1[datecol], r2[datecol]
                if d1 == d2:
                    raise ValueError("Duplicates for lead lags", r1, r2)
                if add1fn(d1) == d2:
                    rs1.append(r2)
                else:
                    cnt = 0
                    d1 = add1fn(d1)
                    while d1 != d2:
                        if cnt >= max_missings:
                            raise ValueError("Too many missings", r1, r2)
                        r0 = empty(r1)
                        r0[datecol] = d1
                        rs1.append(r0)
                        d1 = add1fn(d1)
                        cnt += 1
                    rs1.append(r2)
        else:
            rs1 = rs

        r0 = rs1[0]
        rss = [rs1]
        for n in ns:
            if n >= 0:
                rss.append([empty(r0)] * n + rs1)
            else:
                rss.append(rs1[(-n):] + [empty(r0)] * (-n))

        result = []
        for xs in zip(*rss):
            x0 = xs[0]
            for n, x in zip(ns, xs[1:]):
                for c in cols:
                    strn = str(n) if n >=0 else str(-n) + 'n'
                    x0[c + suffix + strn] = x[c]
            result.append(x0)
        return result
    return fn


def rdate(date):
    return ciso8601.parse_datetime(date)


def datemath(date, n):
    """n means days for "%Y-%m-%d", months for "%Y-%m". Supports only 2 fmt.
    >>> ndate(3)("1993-10")
    '1994-01'
    >>> ndate(-10)("1993-10")
    '1992-12'
    >>> ndate(4)("2012-02-26")
    '2012-03-01'
    >>> ndate(4)("2013-02-26")
    '2013-03-02'
    >>> ndate(-4)("2013-02-26")
    '2013-02-22'
    """
    # "1903-09"
    if len(date) == 7:
        y, m = date.split("-")
        n1 = int(y) * 12 + int(m) + n
        y1, m1 = n1 // 12, n1 % 12
        if m1 == 0:
            y1 -= 1
            m1 = 12
        return str(y1) + '-' + str(m1).zfill(2)
    # "1903-09-29"
    elif len(date) == 10:
        d = ciso8601.parse_datetime(date)
        d1 = d + timedelta(days=n)
        return d1.strftime("%Y-%m-%d")
    else:
        raise ValueError("Unsupported date format", date)


def ols(rs, y, *xs):
    df = pd.DataFrame(rs)
    return sm.OLS(df[[y]], sm.add_constant(df[list(xs)])).fit()


def logit(rs, y, *xs):
    df = pd.DataFrame(rs)
    # disp=0 suppress some infos regarding optimization
    return sm.Logit(df[[y]], sm.add_constant(df[list(xs)])).fit(disp=0)


# no constant version
def ols1(rs, y, *xs):
    df = pd.DataFrame(rs)
    return sm.OLS(df[[y]], df[list(xs)]).fit()


# implicit ordering
def group(rs, key):
    keyfn = _build_keyfn(key)
    rs.sort(key=keyfn)
    return [list(rs1) for _, rs1 in groupby(rs, keyfn)]



def add(**kwargs):
    def fn(r):
        for k, v in kwargs.items():
            try:
                r[k] = v(r)
            except:
                r[k] = ''
        return r
    return fn


#
def addf(**kwargs):
    def fn(rs):
        r0 = rs[0]
        for k, v in kwargs.items():
            try:
                r0[k] = v(rs)
            except:
                r0[k] = ''
        return r0
    return fn


def chunk(rs, n, column=None):
    """
    Usage:
        |  chunk(rs, 3) => returns 3 rows about the same size
        |  chunk(rs, [0.3, 0.4, 0.3]) => returns 3 rows of 30%, 40%, 30%
        |  chunk(rs, [100, 500, 1000], 'col')
        |      => returns 4 rows with break points 100, 500, 1000 of 'col'
    """
    size = len(rs)
    if isinstance(n, int):
        start = 0
        result = []
        for i in range(1, n + 1):
            end = int((size * i) / n)
            # must yield anyway
            result.append(rs[start:end])
            start = end
        return result
    # n is a list of percentiles
    elif not column:
        # then it is a list of percentiles for each chunk
        assert sum(n) <= 1, f"Sum of percentils for chunks must be <= 1.0"
        ns = [int(x * size) for x in accumulate(n)]
        result = []
        for a, b in zip([0] + ns, ns):
            result.append(rs[a:b])
        return result
    # n is a list of break points
    else:
        rs.sort(key=lambda r: r[column])
        start, end = 0, 0
        result = []
        for bp in n:
            while (rs[end][column] < bp) and end < size:
                end += 1
            result.append(rs[start:end])
            start = end
        result.append(rs[end:])
        return result


def winsorize(rs, col, limit=0.01):
    """Winsorsize rows that are out of limits
    Args:
        |  col(str): column name.
        |  limit(float): for both sides respectably.
    returns rs
    """
    xs = [r[col] for r in rs]
    lower = np.percentile(xs, limit * 100)
    higher = np.percentile(xs, (1 - limit) * 100)
    for r in rs:
        if r[col] > higher:
            r[col] = higher
        elif r[col] < lower:
            r[col] = lower
    return rs

def truncate(self, col, limit=0.01):
    """Truncate rows that are out of limits
    Args:
        |  col(str): column name
        |  limit(float): for both sides respectably.
    Returns self
    """
    xs = self[col]
    lower = np.percentile(xs, limit * 100)
    higher = np.percentile(xs, (1 - limit) * 100)
    return self.where(lambda r: r[col] >= lower and r[col] <= higher)


def empty(r):
    r0 = {}
    for c in r:
        r0[c] = ''
    return r0


def isnum(*xs):
    """ Tests if x is numeric
    >>> isnum(3)
    True
    >>> isnum(-29.39)
    True
    >>> isnum('3')
    True
    >>> isnum('-29.39')
    True
    >>> isnum('1,000')
    False
    >>> isnum(3, '1,000')
    False
    >>> isnum(3, '-3.12')
    True
    """
    try:
        for x in xs:
            float(x)
        return True
    except (ValueError, TypeError):
        return False


def stars(pval):
    if pval <= 0.01:
        return "***"
    elif pval <= 0.05:
        return "**"
    elif pval <= 0.10:
        return "*"
    return ""

# ==================================================================================================
# Helpers
# ==================================================================================================

def _build_keyfn(key):
    " if key is a string return a key function "
    # if the key is already a function, just return it
    if hasattr(key, '__call__'):
        return key
    colnames = listify(key)
    # special case
    if colnames == ['*']:
        return lambda r: 1

    if len(colnames) == 1:
        col = colnames[0]
        return lambda r: r[col]
    else:
        return lambda r: [r[colname] for colname in colnames]

def _str2dicts(text):
    f = io.StringIO(text)
    header = next(csv.reader(f))
    return [r for r in csv.DictReader(f, fieldnames=header)]


def listify(x):
    """
    Example:
    >>> listify('a, b, c')
    ['a', 'b', 'c']
    >>> listify(3)
    [3]
    >>> listify([1, 2])
    [1, 2]
    """
    try:
        return [x1.strip() for x1 in x.split(',')]
    except AttributeError:
        try:
            return list(iter(x))
        except TypeError:
            return [x]



# ==================================================================================================
# Backups
# ==================================================================================================
def overlap(rs, size, step=1, key=None):
    if key:
        xs = group(rs, key)
        result = []
        for i in range(0, len(xs), step):
            result.append(list(chain(*xs[i:i + size])))
        return result
    else:
        result = []
        for i in range(0, len(rs), step):
            result.append(rs[i:i + size])
        return result


def simptrans(fname, col1, col2):
    "simple transpose, keep the first column"
    rss = readxl(fname)
    header = [x.strip() for x in next(rss)]
    col0 = header[0]
    for line in rss:
        val0 = line[0]
        for c, val in zip(header[1:], line[1:]):
            yield {
                col0: val0,
                col1: c,
                col2: val
            }


def fnguide(fname, colnames, sheet=None, encoding='euc-kr'):
    rss = readxl(fname, sheet_name=sheet, encoding=encoding)
    for _ in range(8):
        next(rss)
    # firmcodes
    ids = [x[0] for x in grouper(next(rss)[1:], len(colnames))]
    for _ in range(5):
        next(rss)
    header = ['id', 'date'] + colnames
    print(','.join(header))
    for rs in rss:
        date = rs[0]
        for id, vals in zip(ids, grouper(rs[1:], len(colnames))):
            print(','.join([id, str(date)[:10]] + [str(x) for x in vals]))



# There are some easier versions like risk free rates
# def fnguide1(fname, colnames, sheet=None, encoding='euc-kr'):
#     rss = readxl(fname, sheet_name=sheet, encoding=encoding)
#     for _ in range(14):
#         next(rss)
#     for rs in rss:
#         r = {'date': str(rs[0])[:10]}
#         for col, val in zip(colnames, rs[1:]):
#             r[col] = val
#         yield r

def fnguide1(fname, colnames, sheet=None, encoding='euc-kr'):
    rss = readxl(fname, sheet_name=sheet, encoding=encoding)
    for _ in range(14):
        next(rss)
    n = len(colnames)
    print(','.join(['date'] + colnames))
    for rs in rss:
        print(','.join([str(rs[0])[:10]] + [str(x) for x in rs[1:n + 1]]))


def readxl(fname, sheet_name=None, encoding='utf-8'):
    _locale = 'English_United States.1252' if os.name == 'nt' else 'en_US.UTF-8'
    locale.setlocale(locale.LC_ALL, _locale)
    def conv(x):
        try:
            return locale.atoi(x)
        except ValueError:
            try:
                return locale.atof(x)
            except Exception:
                return x

    if fname.endswith('.csv'):
        with open(fname, encoding=encoding) as fin:
            for rs in csv.reader(x.replace('\0', '') for x in fin):
                yield [conv(x) for x in rs]
    else:
        workbook = load_workbook(fname, read_only=True)
        if not sheet_name:
            sheet_name = workbook.sheetnames[0]
        for row in workbook[sheet_name].iter_rows():
            yield [c.value for c in row]


def readcsv(fname):
    with open(fname) as f:
        rss = csv.reader(x.replace('\0', '') for x in f)
        cols = next(rss)
        for rs in rss:
            yield dict(zip(cols, rs))


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def avg(rs, col, wcol=None, ndigits=None):
    if wcol:
        xs = [r for r in rs if isnum(r[col], r[wcol])]
        val = np.average([x[col] for x in xs], weights=[x[wcol] for x in xs]) if xs else ''
    else:
        xs = [r for r in rs if isnum(r[col])]
        val = np.average([x[col] for x in xs]) if xs else ''
    return round(val, ndigits) if ndigits and xs else val


def diff(high, low):
    return [a - b for a, b in zip(high, low)]


def ttest(seq, n=3):
    tval, pval = ttest_1samp(seq, 0.0)
    return f'{round(st.mean(seq), n)}{stars(pval)}', round(tval, n)


def table1d(rs, m, pncol, retcol):
    def get(i):
        return [r for r in rs if r[pncol] == i]

    r = {}
    for i in range(0, m + 1):
        r['p' + str(i)] = avg(get(i), retcol, ndigits=3)
    # diff
    high = [r[retcol] for r in get(m)]
    low = [r[retcol] for r in get(1)]
    v, tval = ttest(diff(high, low))
    r['diff'] = v
    r['tval'] = tval
    yield r


def table2d(rs, m, n, pncol1, pncol2, retcol):
    def get(rs, i, j):
        return [r for r in rs if r[pncol1] == i and r[pncol2] == j]

    for i in range(0, m + 1):
        r = {'p': i}

        for j in range(0, n + 1):
            rs1 = get(rs, i, j)
            r['p' + str(j)] = avg(rs1, retcol, ndigits=3)
        high = [r[retcol] for r in get(rs, i, n)]
        low = [r[retcol] for r in get(rs, i, 1)]

        v, tval = ttest(diff(high, low))
        r['diff'] = v
        r['tval'] = tval
        yield r
    # difference line
    r1 = {'p': 'diff'}
    r2 = {'p': 'tval'}
    for j in range(0, n + 1):
        high = [r[retcol] for r in get(rs, m, j)]
        low = [r[retcol] for r in get(rs, 1, j)]
        v, tval = ttest(diff(high, low))
        r1['p' + str(j)] = v
        r2['p' + str(j)] = tval

    # diff of diff
    hh = [r[retcol] for r in get(rs, m, n)]
    hl = [r[retcol] for r in get(rs, m, 1)]
    lh = [r[retcol] for r in get(rs, 1, n)]
    ll = [r[retcol] for r in get(rs, 1, 1)]

    v, tval = ttest(diff(diff(hh, hl), diff(lh, ll)))

    r1['diff'] = v
    r1['tval'] = tval
    r2['diff'] = ''
    r2['tval'] = ''
    yield r1
    yield r2




def mean(cols, ncol=None):
    def fn(rs):
        r0 = rs[0]
        for c in cols:
            if isinstance(c, str):
                r0[c] = avg(rs, c)
            else:
                # c can be a tuple(for example) for weighted average.
                r0[c[0]] = avg(rs, *c)
        if ncol:
            r0[ncol] = len(rs)
        return r0
    return fn



def portn(n):
    return lambda rs: chunk(rs, n)


def zero_portn(col, n):
    def fn(rs):
        zeros = [r for r in rs if r[col] == 0]
        others = [r for r in rs if r[col] != 0]
        return [zeros] + chunk(others, n)
    return fn


# 1 dimensional numbering
def numbering1(col, port):
    def fn(rs):
        rs = [r for r in rs if isnum(r[col])]
        rs.sort(key=lambda r: r[col])
        for i, rs1 in enumerate(port(rs), 1):
            for r in rs1:
                # you need overall grouping so,
                r['pn_' + col] = 0
                yield r
                r['pn_' + col] = i
                yield r
    return fn


def numbering2(col1, port1, col2, port2):
    def fn(rs):
        rs = [r for r in rs if isnum(r[col1], r[col2])]

        rs.sort(key=lambda r: r[col1])
        for i, rs1 in enumerate(port1(rs), 1):
            for r in rs1:
                r['pn_' + col1] = 0
                r['pn_' + col2] = 0
                yield r
                r['pn_' + col1] = i
                yield r

        rs.sort(key=lambda r: r[col2])
        for j, rs2 in enumerate(port2(rs), 1):
            for r in rs2:
                r['pn_' + col2] = j
                yield r
                r['pn_' + col1] = 0
                yield r

    return fn


def numbering2dep(col1, port1, col2, port2):
    def fn(rs):
        rs = [r for r in rs if isnum(r[col1], r[col2])]
        rs.sort(key=lambda r: r[col1])
        for i, rs1 in enumerate(port1(rs), 1):
            for r in rs1:
                r['pn_' + col1] = 0
                r['pn_' + col2] = 0
                yield r
                r['pn_' + col1] = i
                yield r
            rs1.sort(key=lambda r: r[col2])
            for j, rs2 in enumerate(port2(rs1), 1):
                for r in rs2:
                    r['pn_' + col2] = j
                    yield r
                    r['pn_' + col1] = 0
                    yield r
        yield from rs
    return fn


def append_dummies(r, d):
    for k, v in d.items():
        val = r[k]
        for v1 in v:
            r[k + '_' + str(v1)] = 1 if val == v1 else 0
    return r


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # fnguide('data_for_ff3.xlsx', ['mkt'], sheet='mkt', encoding='utf-8')
    # fnguide('data_for_ff3.xlsx', ['icode'], sheet='icode')
    fnguide('manal.xlsx', ['anal'])
