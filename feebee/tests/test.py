import os
import sys
import unittest

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
    yield r

def sumup1(rs, shipperid):
    if rs[0]['shipperid'] == shipperid:
        r = rs[0] 
        r['norders'] = len(rs)
        yield r


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



class TestProcess(unittest.TestCase):
    def test_example1(self):
        fb.drop('orders')
        fb.process(
            orders = fb.load(file='orders.csv'),

            orders1 = fb.map(
                fn={'yyyymm': lambda r: fb.dconv(r['orderdate'], '%Y-%m-%d', '%Y-%m')},
                data='orders'
            ),

            orders2 = fb.map(fn=sumup, data='orders1', by='yyyymm'),
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
        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('orders_avg_nmonth'):
                if fb.isnum(r['avg']):
                    xs.append(xs)
            self.assertEqual(len(xs), 9)


    def test_example2(self):
        fb.drop('orders, customers')
        fb.process(
            orders=fb.load(file='orders.csv',
                           fn={'yyyymm': lambda r: fb.dconv(r['orderdate'], '%Y-%m-%d', '%Y-%m')}
            ),
            customers=fb.load('customers.csv'),
            orders1=fb.join(
                ['orders', '*', 'customerid'],
                ['customers', 'customername, country', 'CustomerID']
            ),

            orders_by_shippers = fb.map(sumup1, 'orders1', by='shipperid, yyyymm', args=[1, 2, 3])
            
        )

        with fb1._connect('test.db') as c:
            xs = []
            for r in c.fetch('orders_by_shippers'):
                xs.append(r)
            self.assertEqual(len(xs), 24)


if __name__ == "__main__":
    unittest.main()