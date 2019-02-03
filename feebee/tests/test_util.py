import os
import sys
import unittest
import sqlite3

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
from feebee.util import chunk, lag, add_date, read_date, isnum, readxl, grouper, listify

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
        date = rs[0]
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

    


    def tearDown(self):
        remdb()



if __name__ == "__main__":
    unittest.main()


# def addm(date, n):
#     return dmath(date, {'months': n}, '%Y%m')


# # pns (portfolio numbering based on the first date, and assign the same for
# # all the follow up rows, since you hold the portfolio)
# def pns(rs, d, dcol, icol, dep=False):
#     fdate = rs[0][dcol]
#     rs0 = rs.where(f'{dcol}={fdate}')
#     rs1 = rs.where(f'{dcol}!={fdate}')
#     rs0.numbering(d, dep)
#     rs1.follow(rs0, icol, ['pn_' + x for x in list(d)])


# class TestRows(unittest.TestCase):
#     def test_init(self):
#         seq = (Row(x=i) for i in range(3))
#         rs = Rows(seq)
#         self.assertEqual(len(rs.rows), 3)
#         self.assertEqual(len(rs), 3)

#         # No verifications

#     def test_getitem(self):
#         seq = (Row(x=i, y=i + 1) for i in range(3))
#         rs = Rows(seq)
#         self.assertEqual(rs['x'], [0, 1, 2])
#         self.assertEqual(rs['x, y'], [[0, 1], [1, 2], [2, 3]])

#         self.assertEqual(rs[0], rs.rows[0])
#         rs1 = rs[0:2]
#         rs.rows += [Row(x=10, y=11)]
#         self.assertEqual(len(rs1), 2)
#         # rs itself is not changed, shallow copy
#         self.assertEqual(len(rs), 4)
#         rs1[0].x += 1
#         self.assertEqual(rs[0].x, 1)

#     def test_setitem(self):
#         seq = (Row(x=i, y=i + 1) for i in range(3))
#         rs = Rows(seq)
#         rs[1] = 30
#         # non Row is assigned
#         with self.assertRaises(Exception):
#             rs.df()
#         rs[1] = Row(x=10, y=30, z=50)
#         self.assertEqual(rs[1].values, [10, 30, 50])
#         with self.assertRaises(Exception):
#             rs.df()
#         # valid rows is asigned
#         rs[1] = Row(x=10, y=30)
#         self.assertEqual(rs['x'], [0, 10, 2])
#         rs[1:2] = [Row(x=10, y=30)]
#         self.assertEqual(rs['x'], [0, 10, 2])

#         with self.assertRaises(Exception):
#             rs['x'] = ['a', 'b']
#         rs['x'] = ['a', 'b', 'c']

#     def test_delitem(self):
#         seq = (Row(x=i, y=i + 1) for i in range(3))
#         rs = Rows(seq)
#         del rs['x']
#         self.assertEqual(rs[0].columns, ['y'])
#         with self.assertRaises(Exception):
#             del rs['x']
#         del rs['y']
#         self.assertEqual(rs[0].columns, [])

#     def test_isconsec(self):
#         seq = []
#         for i in range(10):
#             seq.append(Row(date=dmath('20010128', {'days': i}, '%Y%m%d')))
#         seq = Rows(seq)
#         self.assertTrue(seq.isconsec('date', '1 day', '%Y%m%d'))
#         del seq[3]
#         self.assertFalse(seq.isconsec('date', '1 day', '%Y%m%d'))

#     def test_roll(self):
#         rs1 = []
#         for year in range(2001, 2011):
#             rs1.append(Row(date=year))

#         lengths = []
#         for rs0 in Rows(rs1).roll(3, 2, 'date', True):
#             lengths.append(len(rs0))
#         self.assertEqual(lengths, [3, 3, 3, 3, 2])

#         lengths = []
#         for rs0 in Rows(rs1).roll(3, 2, 'date'):
#             lengths.append(len(rs0))
#         self.assertEqual(lengths, [3, 3, 3, 3])

#         rs2 = []
#         start_month = '200101'
#         for i in range(36):
#             rs2.append(Row(date=addm(start_month, i)))

#         lengths = []
#         for rs0 in Rows(rs2).where('date > "200103"')\
#                             .roll(12, 12, 'date', lambda d: addm(d, 1), True):
#             lengths.append(len(rs0))
#         self.assertEqual(lengths, [12, 12, 9])

#         lengths = []
#         for rs0 in Rows(rs2).where("date > '200103'")\
#                             .roll(24, 12, 'date', lambda d: addm(d, 1), False):
#             lengths.append(len(rs0))
#         self.assertEqual(lengths, [24])

#         rs3 = []
#         start_date = '20010101'
#         for i in range(30):
#             rs3.append(Row(date=dmath(start_date, {'days': i}, '%Y%m%d')))

#         lengths = []
#         for rs0 in Rows(rs3).roll(14, 7, 'date',
#                                   lambda d: dmath(d, '1 day', '%Y%m%d'), True):
#             lengths.append(len(rs0))
#         self.assertEqual(lengths, [14, 14, 14, 9, 2])

#         # # should be able to handle missing dates
#         rs = Rows([Row(date=addm('200101', i)) for i in range(10)])
#         del rs[3]
#         ls = [[int(x) for x in rs1['date']]
#               for rs1 in rs.roll(5, 4, 'date', lambda d: addm(d, 1), True)]
#         self.assertEqual(
#             ls, [[200101, 200102, 200103, 200105],
#                  [200105, 200106, 200107, 200108, 200109],
#                  [200109, 200110]])

#     def test_order(self):
#         with dbopen('sample.db') as q:
#             seq = (rs[0] for rs in q.fetch('customers', group='country'))
#             q.insert(seq, 'c1')
#             countries = q.rows('c1').order('country', reverse=True)['country']
#             self.assertEqual(len(countries), 21)
#             self.assertEqual(countries[:3], ['Venezuela', 'USA', 'UK'])
#             q.drop('c1')

#     def test_isnum(self):
#         with dbopen('sample.db') as q:
#             rs1 = q.rows('customers', where='isnum(postalcode)')
#             rs2 = q.rows('customers').isnum('postalcode')
#             self.assertEqual(len(rs1), len(rs2))

#             rs1 = q.rows('customers', where='isnum(postalcode, customerid)')
#             rs2 = q.rows('customers', where='isnum(postalcode)')
#             rs3 = q.rows('customers', where='isnum(postalcode, city)')
#             self.assertEqual(len(rs1), 66)
#             self.assertEqual(len(rs2), 66)
#             self.assertEqual(len(rs3), 0)

#     def test_avg(self):
#         with dbopen('sample.db') as q:
#             rs1 = q.rows('products')
#             self.assertEqual(round(rs1.avg('price') * 100), 2887)
#             self.assertEqual(round(rs1.avg('price', 'categoryid') * 100), 2811)

#     def test_ols(self):
#         with dbopen('sample.db') as q:
#             rs = q.rows('products')
#             res = rs.ols('price ~ supplierid + categoryid')
#             self.assertEqual(len(res.params), 3)
#             self.assertEqual(len(res.resid), len(rs))

#     def test_truncate(self):
#         with dbopen('sample.db') as q:
#             rs = q.rows('products')
#             self.assertEqual(len(rs.truncate('price', 0.1)), 61)

#     def test_winsorize(self):
#         with dbopen('sample.db') as q:
#             rs = q.rows('products')
#             self.assertEqual(round(rs.avg('price') * 100), 2887)
#             rs = rs.winsorize('price', 0.2)
#             self.assertEqual(round(rs.avg('price') * 100), 2296)

#     def test_group(self):
#         with dbopen('sample.db') as q:
#             rs = q.rows('customers')
#             ls = []
#             for rs1 in rs.order('country').group('country'):
#                 ls.append(len(rs1))
#             self.assertEqual(ls, [3, 2, 2, 9, 3, 2, 2, 11, 11,
#                                   1, 3, 5, 1, 1, 2, 5, 2, 2, 7, 13, 4])

#     def test_bps(self):
#         rs = Rows(Row(a=i) for i in range(1, 101))
#         self.assertEqual([int(x) for x in rs.bps([0.3, 0.7, 0.8], 'a')],
#                          [30, 70, 80])

#     def test_df(self):
#         with dbopen('sample.db') as q:
#             rs = q.rows('customers')
#             self.assertEqual(rs.df().shape, (91, 7))

#     # pns is a combination of numbering and follow
#     # test numbering and follow
#     def test_numbering(self):
#         with dbopen('sample.db') as c:
#             # now you need yyyy column
#             c.register(lambda d: dconv(d, '%Y-%m-%d', '%Y'), 'yearfn')
#             c.create('select *, yearfn(date) as yyyy from acc1', 'tmpacc1')

#             # oneway sorting
#             c.drop('tmpacc2')
#             for rs in c.fetch('tmpacc1', where='isnum(asset)',
#                               roll=(3, 3, 'yyyy', True)):
#                 pns(rs, {'asset': 10}, dcol='yyyy', icol='id')
#                 c.insert(rs.isnum('pn_asset'), 'tmpacc2')

#             for rs in c.fetch('tmpacc2', roll=(3, 3, 'yyyy', True)):
#                 xs = [len(x) for x in rs.group('yyyy')]
#                 # the first one must have the largest number of items
#                 self.assertEqual(max(xs), xs[0])

#             # average them
#             c.drop('tmpaccavg')
#             for rs in c.fetch('tmpacc2', group='yyyy, pn_asset'):
#                 r = Row()
#                 r.date = rs[0].yyyy
#                 r.pn_asset = rs[0].pn_asset
#                 r.avgasset = rs.avg('asset')
#                 c.insert(r, 'tmpaccavg')

#             # tests if pn numbering is correct!!
#             for rs in c.fetch('tmpaccavg', roll=(3, 3, 'date', True)):
#                 fdate = rs[0]['date']
#                 rs1 = rs.where(f'date={fdate}')
#                 xs1 = rs1.order('pn_asset')['avgasset']
#                 xs2 = rs1.order('avgasset')['avgasset']
#                 self.assertEqual(xs1, xs2)

#             c.drop('tmpacc1, tmpacc2, tmpaccavg')

#     def test_numbering1(self):
#         def fn1(rs):
#             rs0 = rs.where('a=0')
#             rs1 = rs.where('a>0')
#             yield rs0
#             yield rs1

#         rs0 = Rows(Row(a=0) for _ in range(5))
#         for i, r in enumerate(rs0, 1):
#             r.b = i

#         rs1 = Rows(Row(a=1) for _ in range(10))
#         for i, r in enumerate(rs1, 6):
#             r.b = i

#         rs = rs0 + rs1

#         rs['pn_a'] = ''
#         rs['pn_b'] = ''
#         rs.numbering({'a': fn1, 'b': 2}, dep=True)
#         self.assertEqual(rs['pn_a, pn_b'],
#                          [[1, 1], [1, 1],
#                           [1, 2], [1, 2], [1, 2],
#                           [2, 1], [2, 1], [2, 1], [2, 1], [2, 1],
#                           [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

#         rs['pn_a'] = ''
#         rs['pn_b'] = ''
#         rs.numbering({'b': 2, 'a': fn1}, dep=True)
#         self.assertEqual(rs['pn_a, pn_b'],
#                          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
#                           [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
#                           [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

#         rs['pn_a'] = ''
#         rs['pn_b'] = ''
#         rs.numbering({'a': fn1, 'b': 2})
#         self.assertEqual(rs['pn_a, pn_b'],
#                          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
#                           [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
#                           [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

#         rs['pn_b'] = ''
#         rs['pn_a'] = ''
#         rs.numbering({'b': 2, 'a': fn1})
#         self.assertEqual(rs['pn_a, pn_b'],
#                          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
#                           [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
#                           [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

#     def test_numbering2d(self):
#         with dbopen('sample.db') as c:
#             # now you need yyyy column
#             c.register(lambda d: dconv(d, '%Y-%m-%d', '%Y'), 'yearfn')
#             c.create('select *, yearfn(date) as yyyy from acc1', 'tmpacc1')

#             c.drop('tmpacc2')
#             for rs in c.fetch('tmpacc1', where='isnum(asset)',
#                               roll=(8, 8, 'yyyy', True)):
#                 pns(rs, {'asset': 4, 'ppe': 4}, dcol='yyyy', icol='id')
#                 c.insert(rs.isnum('pn_asset, pn_ppe'), 'tmpacc2')

#             import statistics as st
#             for rs in c.fetch('tmpacc2', where='yyyy >= 1988', group='yyyy'):
#                 for i in range(1, 5):
#                     xs = []
#                     for j in range(1, 5):
#                         n = len(rs.where(f'pn_asset={i} and pn_ppe={j}'))
#                         xs.append(n)
#                     self.assertTrue(st.stdev(xs) >= 12)

#             # dependent sort
#             c.drop('tmpacc2')
#             for rs in c.fetch('tmpacc1', where='isnum(asset)',
#                               roll=(8, 8, 'yyyy', True)):
#                 pns(rs, {'asset': 4, 'ppe': 4},
#                     dcol='yyyy', icol='id', dep=True)
#                 c.insert(rs.isnum('pn_asset, pn_ppe'), 'tmpacc2')

#             for rs in c.fetch('tmpacc2', where='yyyy >= 1988', group='yyyy'):
#                 for i in range(1, 5):
#                     xs = []
#                     for j in range(1, 5):
#                         n = len(rs.where(f'pn_asset={i} and pn_ppe={j}'))
#                         xs.append(n)
#                     # number of items ought to be about the same
#                     # Test not so sophisticated
#                     self.assertTrue(st.stdev(xs) < 12)


# # This should be defined in 'main' if you want to exploit multiple cores
# # in Windows, The function itself here is just a giberrish for testing
# def avg_id(rs):
#     r = Row(date=dconv(rs[0].orderdate, '%Y-%m-%d', '%Y%m'))
#     r.orderid = round(rs.avg('orderid'))
#     r.customerid = round(rs.avg('orderid'))
#     r.employeeid = round(rs.avg('employeeid'))
#     r.shipperid = rs[0].shipperid
#     return r


# class TestSQLPlus(unittest.TestCase):
#     # apply is removed but the following works
#     def test_fch(self):
#         with dbopen('sample.db') as c:
#             for r in c.fch('orders'):
#                 self.assertEqual(type(r), sqlite3.Row)
#                 break

#         with dbopen('sample.db') as c:
#             for df in c.fch('orders', group='shipperid'):
#                 self.assertEqual(type(df), pd.core.frame.DataFrame)

#     def test_apply(self):
#         def to_month(r):
#             r.date = dconv(r.orderdate, '%Y-%m-%d', '%Y%m')
#             return r

#         with dbopen('sample.db') as q:
#             tseq = (to_month(r) for r in q.fetch('orders'))
#             q.insert(tseq, 'orders1')

#             ls = []
#             for rs in q.fetch('orders1', group='date'):
#                 ls.append(len(rs))

#             self.assertEqual(ls, [22, 25, 23, 26, 25, 31, 33, 11])
#             self.assertEqual(len(q.rows('orders1')),
#                              sum([22, 25, 23, 26, 25, 31, 33, 11]))

#             ls = []
#             for rs in q.fetch('orders1', roll=(3, 2, 'date', True)):
#                 for rs1 in rs.group('shipperid'):
#                     ls.append(len(rs1))
#             self.assertEqual([sum(ls1) for ls1 in grouper(ls, 3)],
#                              [70, 74, 89, 44])
#             q.drop('orders1')

#     def test_to_csv(self):
#         with dbopen('sample.db') as c:
#             c.to_csv('categories', 'foo.csv')
#             a = c.rows('categories')
#             c.drop('foo')
#             c.load('foo.csv')
#             b = c.rows('foo')
#             for a1, b1 in zip(a, b):
#                 self.assertEqual(a1.values, b1.values)
#             os.remove(os.path.join(getwd(), 'foo.csv'))

#     def test_insert(self):
#         with dbopen('sample.db') as c:
#             c.drop('foo')
#             for rs in c.fetch('orders',  group='shipperid'):
#                 r = rs[0]
#                 r.n = len(rs)
#                 c.insert(r, 'foo')
#             rs = c.rows('foo')
#             self.assertEqual(rs['n'], [54, 74, 68])

#             # the following must not raise exceptions
#             c.insert(Rows([]), 'foo')

#             c.drop('foo')

#             def foo():
#                 for i in range(10):
#                     xs = []
#                     for j in range(3):
#                         xs.append(Row(x=j))
#                     yield Rows(xs)
#             c.insert(foo(), 'foo')
#             self.assertEqual(len(c.rows('foo')), 30)

#             c.drop('foo')

#             def foo():
#                 for i in range(10):
#                     xs = []
#                     for j in range(3):
#                         xs.append(Row(x=j))
#                     yield xs
#             c.insert(foo(), 'foo')
#             self.assertEqual(c.rows('foo')[:6]['x'], [0, 1, 2, 0, 1, 2])

#     def test_register(self):
#         def product(xs):
#             result = 1
#             for x in xs:
#                 result *= x
#             return result

#         with dbopen(':memory:') as c:
#             def foo(x, y):
#                 return x + y

#             def bar(*args):
#                 return sum(args)

#             #
#             def foo1(a, b):
#                 sum = 0
#                 for a1, b1 in zip(a, b):
#                     sum += a1 * b1
#                 return sum

#             def bar1(*args):
#                 sum = 0
#                 for xs in zip(*args):
#                     sum += product(xs)
#                 return sum

#             c.register(foo)
#             c.register(bar)
#             # Look up the def of 'foo1' and you'll see r.a and r.b
#             # Actual table doesn't have to have column a and b
#             c.register_agg(foo1)
#             c.register_agg(bar1)

#             c.sql("create table test(i, j, x)")
#             c.sql("insert into test values (1, 3,'a')")
#             c.sql("insert into test values (21, 2, 'b')")
#             c.sql("insert into test values (5,3, 'a')")
#             c.sql("insert into test values (20,4, 'a')")
#             c.sql("insert into test values (20,'x', 'c')")
#             c.sql("insert into test values (20,-1.2, 'd')")

#             c.create("select foo(i, j) as val1, bar(i, j) as val2 from test",
#                      'test1')
#             self.assertEqual(c.rows('test1')['val1'], [4, 23, 8, 24, '', 18.8])
#             self.assertEqual(c.rows('test1')['val2'], [4, 23, 8, 24, '', 18.8])

#             c.create("""
#             select foo1(i, j) as val1, bar1(i, j) as val2 from test group by x
#             """, 'test2')
#             self.assertEqual(c.rows('test2')['val1'], [98, 42, '', -24.0])
#             self.assertEqual(c.rows('test2')['val2'], [98, 42, '', -24.0])

#     def test_join(self):
#         with dbopen('sample.db') as q:

#             q.join(
#                 ['customers', 'customername', 'customerid'],
#                 # if the matching columns (the third item in the following list
#                 # is missing, then it is assumed to be the same as
#                 # the matching column of the first table
#                 ['orders', 'orderid'],
#                 name='customers1'
#             )
#             rs = q.rows('customers1')
#             self.assertEqual(len(rs), 213)
#             self.assertEqual(len(rs.isnum('orderid')), 196)
#             q.drop('customers1')

#             def to_month(r):
#                 r.date = dconv(r.orderdate, '%Y-%m-%d', '%Y%m')
#                 return r
#             tseq = (to_month(r) for r in q.fetch('orders'))
#             q.insert(tseq, 'orders1', True)
#             # There's no benefits in using multiple cores
#             # You should know what you are doing.

#             tseq = pmap(avg_id, q.fetch('orders1', group='date'),
#                         max_workers=2)
#             q.insert(tseq, 'orders2', True)

#             # testing reel
#             ls = []
#             for rs in q.fetch('orders2', roll=(5, 2, 'date', True)):
#                 ls.append(len(rs))
#             self.assertEqual(ls, [5, 5, 4, 2])

#             self.assertEqual(len(q.rows('orders1')), 196)

#             tseq = (rs[0] for rs in q.fetch('orders1',
#                                             group='date, customerid'))
#             q.insert(tseq, 'orders2', True, pkeys='date, customerid')
#             self.assertEqual(len(q.rows('orders2')), 161)

#             q.register(addm)
#             q.create('select *, addm(date, 1) as d1 from orders1', 'orders1_1')
#             q.create('select *, addm(date, 2) as d2 from orders1', 'orders1_2')
#             q.create('select *, addm(date, 3) as d3 from orders1', 'orders1_3')
#             q.join(
#                 ['orders1', 'date, customerid, orderid', 'date, customerid'],
#                 ['orders1_1', 'orderid as orderid1', 'd1, customerid'],
#                 ['orders1_2', 'orderid as orderid2', 'd2, customerid'],
#                 ['orders1_3', 'orderid as orderid3', 'd3, customerid'],
#                 name='orders3'
#             )
#             q.drop('orders1_1, orders1_2, orders1_3')

#             q.create("""
#             select a.date, a.customerid, a.orderid,
#             b.orderid as orderid1,
#             c.orderid as orderid2,
#             d.orderid as orderid3

#             from orders1 as a

#             left join orders1 as b
#             on a.date = addm(b.date, 1) and a.customerid = b.customerid

#             left join orders1 as c
#             on a.date = addm(c.date, 2) and a.customerid = c.customerid

#             left join orders1 as d
#             on a.date = addm(d.date, 3) and a.customerid = d.customerid
#             """, name='orders4')

#             rs3 = q.rows('orders3')
#             rs4 = q.rows('orders4')

#             for r3, r4 in zip(rs3, rs4):
#                 self.assertEqual(r3.values, r4.values)


# # for pmap, this fn must be in top level, in Windwos machine.
# def fn(r, a):
#     r.x = a
#     return r


# class TestMisc(unittest.TestCase):
#     def test_load_excel(self):
#         with dbopen('sample.db') as c:
#             c.load('orders.xlsx', 'orders_temp')
#             # You may see some surprises because
#             # read_excel uses pandas way of reading excel files
#             # q.rows('orders1').show()
#             self.assertEqual(len(c.rows('orders_temp')), 196)

