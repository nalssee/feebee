import os
import sys
import unittest
import sqlite3
from random import shuffle

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
from feebee.util import chunk, lag, add_date, read_date, isnum, readxl, grouper, listify, truncate,\
    avg, winsorize, ols, group, numbering, affix, where

# only for testing
import feebee.feebee as fb1


def fet(c, tname):
    return list(c.fetch(f"select * from {tname}"))


def remdb():
    if os.path.isfile('test_util.db'):
        os.remove('test_util.db')


def initialize():
    remdb()
    fb1._JOBS = {}
    fb.run()


def ndate(n):
    return lambda date: add_date(date, n)


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


class TestEmAll(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            orderdetails = fb.load('orderdetails.csv'),
            orders = fb.load('orders.csv'),
        )
        fb.run()

    def test_chunk(self):
        rs = [{'x': i} for i in range(10)]

        for rs1 in chunk(rs, 2):
            self.assertEqual(len(rs1), 5)
        # even when there are not enough elements
        ls = chunk(rs, 15)
        self.assertEqual(len(list(ls)), 15)

        # ratio cut
        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'orderdetails')
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
        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'orders')
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

            del rs[2] # raise exception for duplicates, so.
            del rs[-2] # Just for the heck of it
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
        fb.register(
            foreign = fb.load(fnguide('foreign.xlsx', 'buy, sell')),
        )
        # 'foreign' is reserved
        with self.assertRaises(fb1.ReservedKeyword):
            fb.run()
        fb1._JOBS = {}
        fb.register(
            tvol = fb.load(fnguide('foreign.xlsx', 'buy, sell')),
            size = fb.load(fnguide('foreign.xlsx', sheet='size', colnames='size, forsize')),
            mdata = fb.load(fnguide('mdata.csv', colnames='a, b, c, d')),
        )
        fb.run()

        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'tvol')
            self.assertEqual(len(rs), 285888)
            rs = fet(c, 'size')
            self.assertEqual(len(rs), 142944)
            rs = fet(c, 'mdata')
            self.assertEqual(len(rs), 213240)

    def test_listify(self):
        self.assertEqual(listify('a, b, c'), ['a', 'b', 'c'])
        self.assertEqual(listify(3), [3])
        self.assertEqual(listify([1, 2]), [1, 2])

    def test_truncate(self):
        fb.register(
            products = fb.load('products.csv'),
        )
        fb.run()
        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'products')
            self.assertEqual(len(truncate(rs, 'Price', 0.1)), 61)

    def test_winsorize(self):
        fb.register(
            products = fb.load('products.csv'),
        )
        fb.run()

        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'products')
            self.assertEqual(round(avg(rs, 'Price') * 100), 2887)
            rs = winsorize(rs, 'Price', 0.2)
            self.assertEqual(round(avg(rs, 'Price') * 100), 2296)

    def test_avg(self):
        fb.register(
            products = fb.load('products.csv'),
        )
        fb.run()

        with fb1._connect('test_util.db') as c:
            rs1 = fet(c, 'products')
            self.assertEqual(round(avg(rs1, 'Price') * 100), 2887)
            self.assertEqual(round(avg(rs1, 'Price', 'CategoryID') * 100), 2811)

    def test_ols(self):
        fb.register(
            products = fb.load('products.csv'),
        )
        fb.run()

        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'products')
            res = ols(rs, 'Price', 'SupplierID, CategoryID')
            self.assertEqual(len(res.params), 3)
            self.assertEqual(len(res.resid), len(rs))
            # no constant
            res = ols(rs, 'Price', 'SupplierID, CategoryID', False)
            self.assertEqual(len(res.params), 2)
            self.assertEqual(len(res.resid), len(rs))

    def test_group(self):
        fb.register(
            customers = fb.load('customers.csv'),
        )
        fb.run()

        with fb1._connect('test_util.db') as c:
            rs = fet(c, 'customers')
            ls = [len(g) for g in group(rs, 'Country')]
            self.assertEqual(ls, [3, 2, 2, 9, 3, 2, 2, 11, 11,
                                  1, 3, 5, 1, 1, 2, 5, 2, 2, 7, 13, 4])

    def test_numbering(self):
        rs = [{'a': i // 2 if i %  2 == 0 else '', 'b': i // 3 if i % 3 == 0 else ''} for i in range(1, 101)]
        shuffle(rs)
        numbering({'a': 2, 'b': 3})(rs)
        rs = [r for r in rs if isnum(r['a'], r['b'])]
        rs.sort(key=lambda r: r['a'])
        self.assertEqual([r['pn_b'] for r in rs],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        )

        rs = [{'a': i // 2 if i %  2 == 0 else '', 'b': i // 3 if i % 3 == 0 else ''} for i in range(1, 101)]
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


    def tearDown(self):
        remdb()


if __name__ == "__main__":
    unittest.main()