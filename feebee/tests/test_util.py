import os
import sys
import unittest
from itertools import groupby
from random import shuffle

import mapon.mapon as mo
import numpy as np
import statsmodels.api as sm
from mapon.config import _DBNAME
from mapon.util import (add_date, allnum, avg, chunk, getX, gety, group, isnum,
                        lag, listify, overlap, readxl, set_default, step,
                        where)
from more_itertools import grouper


def remdb():
    if os.path.isfile(_DBNAME):
        os.remove(_DBNAME)


def initialize():
    remdb()
    mo._JOBS = {}
    mo.run()


def ndate(n):
    return lambda date: add_date(date, n)


# some of the depricated functions
def fnguide(fname, colnames, sheet=None, encoding='euc-kr'):
    colnames = listify(colnames)
    ncols = len(colnames)
    rss = readxl(fname, sheets=sheet, encoding=encoding)
    for _ in range(8):
        next(rss)
    # firmcodes
    ids = [x[0] for x in grouper(next(rss)[1:], ncols)]
    for _ in range(5):
        next(rss)

    for rs in rss:
        date = str(rs[0])[:10]
        for id, vals in zip(ids, grouper(rs[1:], ncols)):
            yield {'id': id, 'date': date, **{c: v for c, v in zip(colnames, vals)}}


def _num1(rs, c, p):
    rs = [r for r in rs if isnum(r[c])]
    rs.sort(key=lambda r: r[c])
    rss = chunk(rs, p) if isinstance(p, int) else p(rs)
    for i, rs1 in enumerate(rss, 1):
        for r in rs1:
            r['pn_' + c] = i
    return rss


def numbering(d, dep=False):
    def fni(rs, cps):
        if cps:
            c, p = cps[0]
            _num1(rs, c, p)
            fni(rs, cps[1:])
        return rs

    def fnd(rs, cps):
        if cps:
            c, p = cps[0]
            for rs1 in _num1(rs, c, p):
                fnd(rs1, cps[1:])
        return rs

    cps = [(c, p) for c, p in d.items()]

    def fn(rs):
        for c, _ in cps:
            for r in rs:
                r['pn_' + c] = ''
        return fnd(rs, cps) if dep else fni(rs, cps)
    return fn


def winsorize(rs, col, limit=0.01):
    """Winsorsize rows that are out of limits
    Args:
        |  col(str): column name.
        |  limit(float): for both sides respectably.
    returns rs
    """
    rs = [r for r in rs if isnum(r[col])]
    xs = [r[col] for r in rs]
    lower = np.percentile(xs, limit * 100)
    higher = np.percentile(xs, (1 - limit) * 100)
    for r in rs:
        if r[col] > higher:
            r[col] = higher
        elif r[col] < lower:
            r[col] = lower
    return rs


def truncate(rs, col, limit=0.01):
    """Truncate rows that are out of limits
    Args:
        |  col(str): column name
        |  limit(float): for both sides respectably.
    Returns self
    """
    rs = [r for r in rs if isnum(r[col])]
    xs = [r[col] for r in rs]
    lower = np.percentile(xs, limit * 100)
    higher = np.percentile(xs, (1 - limit) * 100)
    return [r for r in rs if r[col] >= lower and r[col] <= higher]


def affix(**kwargs):
    def fn(r):
        for k, v in kwargs.items():
            try:
                r[k] = v(r)
            except:
                r[k] = ''
        return r
    return fn


class TestEmAll(unittest.TestCase):
    def setUp(self):
        initialize()
        mo.register(
            orderdetails=mo.load('orderdetails.csv'),
            orders=mo.load('orders.csv'),
        )
        mo.run()

    def test_chunk(self):
        rs = [{'x': i} for i in range(10)]

        for rs1 in chunk(rs, 2):
            self.assertEqual(len(rs1), 5)
        # even when there are not enough elements
        ls = chunk(rs, 15)
        self.assertEqual(len(list(ls)), 15)

        # ratio cut
        with mo._connect(_DBNAME) as c:
            rs = mo.get('orderdetails')
            a, b, c = chunk(rs, [0.3, 0.4, 0.3])
            n = len(rs)
            self.assertEqual(len(a), int(n * 0.3))
            self.assertEqual(len(b), int(n * 0.4))
            # roughly the same
            self.assertEqual(len(c), int(n * 0.3) + 1)

        rs = [{'a': i} for i in [1, 7, 3, 7]]
        # break point
        xs = [[x['a'] for x in xs] for xs in chunk(rs, [2, 5], 'a')]
        self.assertEqual(xs, [[1], [3], [7, 7]])

        xs = [[x['a'] for x in xs] for xs in chunk(rs, [2, 2.5], 'a')]
        self.assertEqual(xs, [[1], [], [3, 7, 7]])

        xs = [[x['a'] for x in xs] for xs in chunk(rs, [1, 3, 5], 'a')]
        self.assertEqual(xs, [[], [1], [3], [7, 7]])

    def test_lag(self):
        with mo._connect(_DBNAME) as c:
            rs = mo.get( 'orders')
            result = lag('orderid, customerid', 'orderdate', [1, 2, -1])(rs)
            self.assertEqual(len(rs), len(result))
            cols = list(result[0].keys())
            self.assertEqual(cols,
                ['orderid', 'customerid', 'employeeid', 'orderdate', 'shipperid', 'orderid_1', 'customerid_1', 'orderid_2', 'customerid_2', 'orderid_1n', 'customerid_1n'])

            xs = [r['orderid'] for r in result]
            self.assertEqual(xs[:7], [10248, 10249, 10250, 10251, 10252, 10253, 10254])

            ys = [r['orderid_2'] for r in result]
            self.assertEqual(ys[2:], xs[:-2])
            self.assertEqual(ys[:2], ['', ''])

            zs = [r['orderid_1n'] for r in result]
            self.assertEqual(xs[1:], zs[:-1])
            self.assertEqual(zs[-1:], [''])
            rs = rs[:7]
            with self.assertRaises(ValueError):
                result = lag('orderid, customerid', 'orderdate', [1, 2, -1], ndate(1))(rs)

            del rs[2]  # raise exception for duplicates, so.
            del rs[-2]  # Just for the heck of it
            result = lag('orderid, customerid', 'orderdate', [1, 2, -1], ndate(1))(rs)

            self.assertEqual([r['orderdate'] for r in result],
                             ['1996-07-04', '1996-07-05', '1996-07-06', '1996-07-07',
                              '1996-07-08', '1996-07-09', '1996-07-10', '1996-07-11'])
            self.assertEqual(
                [r['orderid'] for r in result],
                [10248, 10249, '', '', 10251, 10252, '', 10254]
            )
            self.assertEqual(
                [r['orderid_2'] for r in result],
                ['', '', 10248, 10249, '', '', 10251, 10252]
            )
            self.assertEqual(
                [r['orderid_1n'] for r in result],
                [10249, '', '', 10251, 10252, '', 10254, '']
            )

    def test_add_date(self):
        self.assertEqual(add_date('1993-10', 3), '1994-01')
        self.assertEqual(add_date('1993-10', -10), '1992-12')
        self.assertEqual(add_date('2012-02-26', 4), '2012-03-01')
        self.assertEqual(add_date('2013-02-26', 4), '2013-03-02')
        self.assertEqual(add_date('2013-02-26', -4), '2013-02-22')

    def test_isnum(self):
        self.assertTrue(isnum(3))
        self.assertTrue(isnum(-29.39))
        self.assertTrue(isnum('3'))
        self.assertTrue(isnum('-29.39'))
        self.assertFalse(isnum('1,000'))
        self.assertFalse(isnum(3, '1,000'))
        self.assertTrue(isnum(3, '-3.12'))

    def test_readxl(self):
        mo.register(
            foreign=mo.load(fnguide('foreign.xlsx', 'buy, sell')),
        )
        # 'foreign' is reserved
        with self.assertRaises(mo.ReservedKeyword):
            mo.run()
        mo._JOBS = {}
        mo.register(
            tvol=mo.load(fnguide('foreign.xlsx', 'buy, sell')),
            size=mo.load(fnguide('foreign.xlsx', sheet='size', colnames='size, forsize')),
            mdata=mo.load(fnguide('mdata.csv', colnames='a, b, c, d')),
        )
        mo.run()

        with mo._connect(_DBNAME) as c:
            rs = mo.get('tvol')
            self.assertEqual(len(rs), 285888)
            rs = mo.get('size')
            self.assertEqual(len(rs), 142944)
            rs = mo.get('mdata')
            self.assertEqual(len(rs), 213240)

    def test_readxl2(self):
        for sheet, rs in readxl('foreign.xlsx', sheets='*', by_sheet=True):
            if sheet == 'tvol':
                for _ in range(14):
                    next(rs)
                self.assertEqual(next(rs)[1], 3810614)
            elif sheet == 'size':
                for _ in range(17):
                    next(rs)
                self.assertEqual(next(rs)[1], 13290693)


    # xls file reading
    def test_readxl3(self):
        self.assertEqual(len(list(readxl('ff.xls'))), 961)

    def test_listify(self):
        self.assertEqual(listify('a, b, c'), ['a', 'b', 'c'])
        # return as is
        self.assertEqual(listify(3), 3)
        self.assertEqual(listify([1, 2]), [1, 2])

    def test_truncate(self):
        mo.register(
            products=mo.load('products.csv'),
        )
        mo.run()
        with mo._connect(_DBNAME) as c:
            rs = mo.get('products')
            self.assertEqual(len(truncate(rs, 'Price', 0.1)), 61)

    def test_winsorize(self):
        mo.register(
            products=mo.load('products.csv'),
        )
        mo.run()

        with mo._connect(_DBNAME) as c:
            rs = mo.get('products')
            self.assertEqual(round(avg(rs, 'Price') * 100), 2887)
            rs = winsorize(rs, 'Price', 0.2)
            self.assertEqual(round(avg(rs, 'Price') * 100), 2296)

    def test_avg(self):
        mo.register(
            products=mo.load('products.csv'),
        )
        mo.run()

        with mo._connect(_DBNAME) as c:
            rs1 = mo.get('products')
            self.assertEqual(round(avg(rs1, 'Price') * 100), 2887)
            self.assertEqual(round(avg(rs1, 'Price', 'CategoryID') * 100), 2811)

    def test_ols(self):
        mo.register(
            products=mo.load('products.csv'),
        )
        mo.run()

        with mo._connect(_DBNAME) as c:
            rs = mo.get('products')
            # with Constant
            y, X = gety(rs, 'Price'), getX(rs, 'SupplierID, CategoryID', True)
            res = sm.OLS(y, X).fit()
            self.assertEqual(len(res.params), 3)
            self.assertEqual(len(res.resid), len(rs))
            # no constant
            y, X = gety(rs, 'Price'), getX(rs, 'SupplierID, CategoryID')
            res = sm.OLS(y, X).fit()
            self.assertEqual(len(res.params), 2)
            self.assertEqual(len(res.resid), len(rs))

    def test_group(self):
        mo.register(
            customers=mo.load('customers.csv'),
        )
        mo.run()

        with mo._connect(_DBNAME) as c:
            rs = mo.get('customers')
            ls = [len(g) for g in group(rs, 'Country')]
            self.assertEqual(ls, [3, 2, 2, 9, 3, 2, 2, 11, 11,
                                  1, 3, 5, 1, 1, 2, 5, 2, 2, 7, 13, 4])

    def test_numbering(self):
        rs = [{'a': i // 2 if i % 2 == 0 else '', 'b': i // 3 if i % 3 == 0 else ''} for i in range(1, 101)]
        shuffle(rs)
        numbering({'a': 2, 'b': 3})(rs)
        rs = [r for r in rs if isnum(r['a'], r['b'])]
        rs.sort(key=lambda r: r['a'])
        self.assertEqual([r['pn_b'] for r in rs],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        )

        rs = [{'a': i // 2 if i % 2 == 0 else '', 'b': i // 3 if i % 3 == 0 else ''} for i in range(1, 101)]
        shuffle(rs)
        numbering({'a': 2, 'b': 3}, True)(rs)
        rs = [r for r in rs if isnum(r['a'], r['b'])]
        rs.sort(key=lambda r: r['a'])
        self.assertEqual([r['pn_b'] for r in rs if isnum(r['a'], r['b'])],
            [1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 3, 3, 3]
        )

    def test_affix(self):
        rs = [{'a': i} for i in range(3)]
        fn = affix(b=lambda r: r['a'] + 1, c=lambda r: r['a'] + 2)
        rs = [fn(r) for r in rs]
        self.assertEqual(
            [r['b'] for r in rs],
            [1, 2, 3]
        )
        self.assertEqual(
            [r['c'] for r in rs],
            [2, 3, 4]
        )

    def test_where(self):
        rs = [{'a': i} for i in range(10)]
        fn = where(lambda r: r['a'] > 7, lambda rs: rs[0])
        self.assertEqual(
            fn(rs)['a'], 8
        )

        fn = where(lambda r: r['a'] > 7)
        self.assertEqual(
            [fn(r) for r in rs][:7], [None] * 7
        )

    def test_overlap(self):
        rs = [{'a': i} for i in range(10)]
        rss = overlap(rs, 5, 3)
        self.assertEqual([[r['a'] for r in rs1] for rs1 in rss],
                         [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 9], [9]])

        with mo._connect(_DBNAME) as c:
            rs = []
            for r in mo.get('orders'):
                r['yyyymm'] = r['orderdate'][:7]
                rs.append(r)

            shuffle(rs)
            rss = overlap(rs, 3, key='yyyymm')
            self.assertEqual(
                [rs1[0]['yyyymm'] for rs1 in rss],
                [add_date('1996-07', i) for i in range(8)])
            self.assertEqual(
                [rs1[-1]['yyyymm'] for rs1 in rss],
                ['1996-09', '1996-10', '1996-11', '1996-12', '1997-01', '1997-02', '1997-02', '1997-02'])
            self.assertEqual(
                [len(rs1) for rs1 in rss],
                [70, 74, 74, 82, 89, 75, 44, 11]
            )

    def test_set_default(self):
        rs = [
            {'a': 10},
            {'a': 11},
            {'a': -3}
        ]

        set_default(rs, 'b, c')
        self.assertEqual([3, 3, 3], [len(r)for r in rs])

    def test_allnum(self):
        rs = [
            {'a': -7.9, 'b': 'a'},
            {'a': '', 'b': -3},
            {'a': -3, 'b': 1.2}
        ]
        self.assertEqual(len(allnum(rs, 'a, b')), 1)
        self.assertEqual(len(allnum(rs, 'a')), 2)
        self.assertEqual(len(allnum(rs, 'b')), 2)

    def test_step(self):
        rs1 = [
            {'a': 10},
            {'a': -3},
            {'a': 12},
        ]

        rs2 = [
            {'a': 12},
            {'a': 3},
            {'a': 30},
            {'a': 10},
        ]

        key1 = lambda r: r['a']
        rs1.sort(key=key1)
        rs2.sort(key=key1)
        rs1 = groupby(rs1, key1)
        rs2 = groupby(rs2, key1)
        result = []
        for a, b in step([rs1, rs2]):
            result.append((a[0]['a'] if a else None, b[0]['a'] if b else None))
        self.assertEqual(result, [(-3, None), (None, 3), (10, 10), (12, 12), (None, 30)])

    def tearDown(self):
        remdb()


if __name__ == "__main__":
    unittest.main()
