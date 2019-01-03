import os
import sys
import unittest
from datetime import datetime
from dateutil.relativedelta import relativedelta


TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
# only for testing
import feebee.feebee as fb1

# customers.csv
# CustomerID,CustomerName,ContactName,Address,City,PostalCode,Country
# 1,Alfreds Futterkiste,Maria Anders,Obere Str. 57,Berlin,12209,Germany
# 2,Ana Trujillo Emparedados y helados,Ana Trujillo,Avda. de la Constitución 2222,México D.F.,5021,Mexico
# 3,Antonio Moreno Taquería,Antonio Moreno,Mataderos 2312,México D.F.,5023,Mexico
# 4,Around the Horn,Thomas Hardy,120 Hanover Sq.,London,WA1 1DP,UK
# 5,Berglunds snabbköp,Christina Berglund,Berguvsvägen 8,Luleå,S-958 22,Sweden


# orders.csv 파일은 대략 이렇게 생겼다
# orderid,customerid,employeeid,orderdate,shipperid
# 10248,90,5,1996-07-04,3
# 10249,81,6,1996-07-05,1
# 10250,34,4,1996-07-08,2
# 10251,84,3,1996-07-08,1

def sumup(rs):
    r = rs[0]
    r['norders'] = len(rs)
    return r

def bigmarket(rs, a):
    if len(rs) > a:
        yield from rs

def bigmarket1(rs, a):
    if len(rs) > a:
        return rs
    else:
        return []


def cnt(rs):
    for i, r in enumerate(rs):
        r['cnt'] = i
        yield r


def orders_avg_nmonth(r):
    r['nmonth'] = 3
    try:
        r['avg'] = round((r['norders'] + r['norders1'] + r['norders2']) / 3, 1)
    except:
        r['avg'] = ''
    yield r

    r['nmonth'] = 6
    try:
        r['avg'] = round((r['norders'] + r['norders1'] + r['norders2'] +  \
                          r['norders3'] + r['norders4'] + r['norders5']) / 6, 1)
    except:
        r['avg'] = ''
    yield r


def add(**kwargs):
    def fn(r):
        for k, v in kwargs.items():
            r[k] = v(r)
        return r
    return fn


def dconv(date, infmt, outfmt=None, **size):
    """Date arithmetic
    Returns int if input(date) is int else str
    """
    outfmt = outfmt or infmt
    if not size:
        # Just convert the format
        return datetime.strftime(datetime.strptime(str(date), infmt), outfmt)
    d1 = datetime.strptime(str(date), infmt) + relativedelta(**size)
    d2 = d1.strftime(outfmt)
    return int(d2) if isinstance(date, int) else d2

def isnum(*xs):
    "Tests if x is numeric"
    try:
        for x in xs:
            float(x)
        return True
    except (ValueError, TypeError):
        return False

def errornous1(rs):
    r = rs[0]
    yield r
    r['x'] = 10
    yield r

def errornous2(r):
    if r['customerid'] > 1000:
        yield r


def add1(rs, col):
    return rs[0][col] + 1

class TestProcess(unittest.TestCase):
    def test_example1(self):
        with fb1._connect('test.db') as c:
            c.drop('orders, customers')

        fb1._JOBS = {}
        fb.register(
            orders = fb.load(file='orders.csv'),

            orders1 = fb.map(
                fn=add(yyyymm=lambda r: dconv(r['orderdate'], '%Y-%m-%d', '%Y-%m')),
                data='orders'
            ),

            orders2 = fb.map(fn=sumup, data='orders1', by='yyyymm', parallel=True),
            orders3 = fb.map(fn=cnt, data='orders2', by='*'),

            orders4 = fb.join(
                ['orders3', '*', 'cnt'],
                ['orders3', 'norders as norders1', 'cnt + 1'],
                ['orders3', 'norders as norders2', 'cnt + 2'],
                ['orders3', 'norders as norders3', 'cnt + 3'],
                ['orders3', 'norders as norders4', 'cnt + 4'],
                ['orders3', 'norders as norders5', 'cnt + 5']
            ),

            orders_avg_nmonth = fb.map(fn=orders_avg_nmonth, data='orders4'),
        )

        fb.run()

        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('orders_avg_nmonth'):
                if isnum(r['avg']):
                    xs.append(xs)
            self.assertEqual(len(xs), 9)

    def test_example2(self):
        with fb1._connect('test.db') as c:
            c.drop('orders, customers')

        fb1._JOBS = {}
        fb.register(
            customers = fb.load(file='customers.csv'),
            customers1 = fb.map(bigmarket, 'customers', by='Country', arg=5),
            # 2 is a chunksize, default is 1
            customers2 = fb.map(bigmarket1, 'customers', by='Country', arg=5, parallel=2)
        )
        fb.run()

        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('customers1'):
                xs.append(r)

            xs2 = []
            for r in c.fetch('customers2'):
                xs2.append(r)
            self.assertEqual(len(xs), len(xs2))


    # union
    def test_example3(self):
        with fb1._connect('test.db') as c:
            c.drop('orders')

        fb1._JOBS = {}
        fb.register(
            orders=fb.load(file='orders.csv'),
            orders1=fb.map(lambda r: r, 'orders'),
            orders2=fb.union('orders, orders1')
            # the following is also fine
            # orders2=fb.union(['orders', 'orders1'])
        )
        fb.run()

        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('orders'):
                xs.append(r)

            xs2 = []
            for r in c.fetch('orders2'):
                xs2.append(r)

            self.assertEqual(len(xs) * 2, len(xs2))

    def test_example4(self):
        with fb1._connect('test.db') as c:
            c.drop('orders')

        fb1._JOBS = {}
        fb.register(
            orders=fb.load(file='orders.csv'),
        )
        # You can do something similar to macro programming
        # by separating registration. Imagine
        fb.register(
            orders1=fb.map(lambda r: r, 'orders'),
            orders2=fb.union('orders, orders1')
        )
        fb.run()

        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('orders'):
                xs.append(r)

            xs2 = []
            for r in c.fetch('orders2'):
                xs2.append(r)

            self.assertEqual(len(xs) * 2, len(xs2))


    def test_error1(self):
        with fb1._connect('test.db') as c:
            c.drop('orders')

        fb1._JOBS = {}
        fb.register(
            orders=fb.load(file='orders.csv'),
            orders1=fb.map(errornous1, 'orders', by='*')
        )
        jobs = fb.run()

        self.assertEqual([j['output'] for j in jobs], ['orders1'])

    def test_error2(self):
        with fb1._connect('test.db') as c:
            c.drop('orders')

        fb1._JOBS = {}
        fb.register(
            orders=fb.load(file='orders.csv'),
            orders1=fb.map(errornous2, 'orders')
        )
        jobs = fb.run()
        self.assertEqual([j['output'] for j in jobs], ['orders1'])

    def test_error3(self):
        fb.register(
            _temp = fb.map(lambda r: r, 'very_unlikely_table_name')
        )
        fb.run()



if __name__ == "__main__":
    unittest.main()