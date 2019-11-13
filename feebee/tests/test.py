import os
import sys
import unittest
import sqlite3
from itertools import groupby

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
# only for testing
import feebee.feebee as fb1
from feebee.util import step, head

# customers.csv
# CustomerID,CustomerName,ContactName,Address,City,PostalCode,Country
# 1,Alfreds Futterkiste,Maria Anders,Obere Str. 57,Berlin,12209,Germany
# 2,Ana Trujillo Emparedados y helados,Ana Trujillo,Avda. de la Constitución 2222,México D.F.,5021,Mexico
# 3,Antonio Moreno Taquería,Antonio Moreno,Mataderos 2312,México D.F.,5023,Mexico
# 4,Around the Horn,Thomas Hardy,120 Hanover Sq.,London,WA1 1DP,UK
# 5,Berglunds snabbköp,Christina Berglund,Berguvsvägen 8,Luleå,S-958 22,Sweden


# orders.csv
# orderid,customerid,employeeid,orderdate,shipperid
# 10248,90,5,1996-07-04,3
# 10249,81,6,1996-07-05,1
# 10250,34,4,1996-07-08,2
# 10251,84,3,1996-07-08,1


def remdb():
    if os.path.isfile('test.db'):
        os.remove('test.db')


def initialize():
    remdb()
    fb1._JOBS = {}
    fb.run()


class TestLoading(unittest.TestCase):
    def setUp(self):
        initialize()
        # make empty table

    def test_dbfilename_created_as_script_name(self):
        # fb.run in setUp created empty test.db
        name, _ = os.path.splitext(os.path.basename(__file__))
        self.assertIn(name + '.db', os.listdir())

    def test_loading_ordinary_csv(self):
        fb.register(orders=fb.load('orders.csv'))
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(list(fet(c, 'orders'))), nlines_file('orders.csv') - 1)

    # TODO: some of the other options like encoding must be tested
    def test_loading_ordinary_tsv(self):
        fb.register(markit=fb.load('markit.tsv'))
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(list(fet(c, 'markit'))), nlines_file('markit.tsv') - 1)

    def test_loading_semicolon_separated_file(self):
        fb.register(orders1=fb.load('orders1.txt', delimiter=";"))
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(list(fet(c, 'orders1'))), nlines_file('orders1.txt') - 1)
            with open('orders1.txt') as f:
                self.assertEqual(f.readline()[:-1].split(";"),
                                 c._cols('select * from orders1'))

    def test_loading_excel_files(self):
        fb.register(
            ff=fb.load('ff.xls'),
            ff1=fb.load('ff.xlsx'),
            na_sample=fb.load('na_sample.xlsx'),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(set(c.get_tables()), set(['ff', 'ff1', 'na_sample']))

            for line in fet(c, 'na_sample'):
                print(line)


    def test_loading_sas_file(self):
        fb.register(
            ff5=fb.load('ff5_ew_mine.sas7bdat'),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(set(c.get_tables()), set(['ff5']))

    def test_loading_stata_file(self):
        fb.register(
            crime=fb.load('crime.dta'),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(list(fet(c, 'crime'))), 2725)

    def test_loading_seq(self):
        def add3(r):
            r['b'] = r['a'] + 3
            return r

        fb.register(
            foo=fb.load(({'a': i} for i in range(10)), fn=add3)
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(c._cols("select * from foo"), ['a', 'b'])

    def tearDown(self):
        if os.path.isfile('test.db'):
            os.remove('test.db')


class TestNone(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            vendors=fb.load('tysql_Vendors.csv'),
        )
        fb.run()

    def test_none(self):
        def input_none(r):
            if not r['vend_state']:
                r['vend_state'] = None
            return r

        # inserting string 'None' doesn't actually make it None
        def input_none1(r):
            if not r['vend_state']:
                r['vend_state'] = 'None'
            return r

        fb.register(
            vendors1=fb.map(input_none, 'vendors'),
            vendors2=fb.map(input_none1, 'vendors'),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(len([r for r in fet(c, 'vendors') if r['vend_state'] == '  ']), 1)
            self.assertEqual(len([r for r in fet(c, 'vendors') if r['vend_state'] is '']), 1)
            # '' is falsy but '  ' is not
            self.assertEqual(len([r for r in fet(c, 'vendors1') if r['vend_state'] is None]), 1)
            self.assertEqual(len([r for r in fet(c, 'vendors2') if r['vend_state'] is None]), 0)

    def tearDown(self):
        if os.path.isfile('test.db'):
            os.remove('test.db')


class TestMap(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            orders=fb.load('orders.csv'),
            customers=fb.load('customers.csv'),
        )
        fb.run()
    
    def test_arg(self):
        def sizefn(r, ff):
            r['n'] = len(ff)
            yield r
        fb.register(
            orders1 = fb.map(sizefn, 'orders', arg=fb.read('ff')),
            orders2 = fb.map(sizefn, 'orders', arg=fb.read('ff'), parallel=True),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            xs1 = fet(c, 'orders1')
            self.assertEqual([r['n'] for r in xs1], [960] * len(xs1))
            self.assertEqual(xs1, fet(c, 'orders2'))

    def test_append_yyyy_yyyymm_columns(self):
        def add_yyyy_yyyymm(r):
            if r['orderdate'] > '1996-xx-xx':
                r['yyyy'] = r['orderdate'][:4]
                r['yyyymm'] = r['orderdate'][:7]
                yield r

        fb.register(
            orders1=fb.map(add_yyyy_yyyymm, 'orders'),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(c._cols('select * from orders')) + 2,
                             len(c._cols('select * from orders1')))
            self.assertEqual(sum(1 for _ in fet(c, 'orders1')), 44)

    def test_group_by(self):
        def bigmarket(a):
            def fn(rs):
                if len(rs) > a:
                    yield from rs
            return fn

        def bigmarket1(a):
            def fn(rs):
                if len(rs) > a:
                    return rs
                else:
                    return []
            return fn

        fb.register(
            # you can either pass a function that returns
            # a dictionary (row) or  a list of dictionaries
            # or pass a generator that yields dictionaries
            customers1=fb.map(bigmarket(5), 'customers', by='Country'),
            customers2=fb.map(bigmarket1(5), 'customers', by='Country')
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(list(fet(c, 'customers1')), list(fet(c, 'customers2')))

    def test_group_n(self):
        fb.register(
            orders1=fb.map(lambda rs: {'a': len(rs)}, 'orders', by=10)
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(fet(c, 'orders')), sum(r['a'] for r in fet(c, 'orders1')))

    # all of them at once
    def test_group_star(self):
        fb.register(
            orders1=fb.map(lambda rs: rs, 'orders', by=' * '),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(fet(c, 'orders'), fet(c, 'orders1'))

    def test_group_invalid(self):
        fb.register(
            orders1=fb.map(lambda rs: {'a': 10}, 'orders', by=10.0),
        )
        _, undone = fb.run()
        self.assertEqual([x['output'] for x in undone], ['orders1'])

    def test_insert_empty_rows(self):
        def filter10(r):
            if r['shipperid'] == 10:
                yield r

        fb.register(
            orders1=fb.map(filter10, 'orders')
        )
        fb.run()
        with self.assertRaises(sqlite3.OperationalError):
            with fb1._connect('test.db') as c:
                list(fet(c, 'orders1'))

    def test_return_none(self):
        def foo(r):
            if r['shipperid'] == 1:
                return r

        fb.register(
            orders1=fb.map(foo, 'orders'),
        )
        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(len(list(fet(c, 'orders1'))), 54)

    def tearDown(self):
        remdb()


class TestMapErrornousInsertion(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            orders=fb.load(file='orders.csv'),
        )
        fb.run()

    def test_insert_differently_named_rows(self):
        def errornous1(rs):
            r = rs[0]
            yield r
            del r['customerid']
            r['x'] = 10
            yield r

        fb.register(orders1=fb.map(errornous1, 'orders', by='*'))
        fb.run()
        # orders1 is not created
        with self.assertRaises(sqlite3.OperationalError):
            with fb1._connect('test.db') as c:
                list(fet(c, 'orders1'))

    def test_insert_1col_deleted(self):
        def errornous1(rs):
            r = rs[0]
            yield r
            del r['customerid']
            yield r

        fb.register(orders1=fb.map(errornous1, 'orders', by='*'))
        fb.run()
        # orders1 is not created
        with self.assertRaises(sqlite3.OperationalError):
            with fb1._connect('test.db') as c:
                for r in fet(c, 'orders1'):
                    print(r)
                list(fet(c, 'orders1'))

    def test_insert_1col_added(self):
        def errornous1(rs):
            r = rs[0]
            yield r
            # added 'xxx' col is ignored
            r['xxx'] = 10
            yield r

        fb.register(orders1=fb.map(errornous1, 'orders', by='*'))
        fb.run()
        with fb1._connect('test.db') as c:
            x1, x2 = list(fet(c, 'orders1'))
            self.assertEqual(x1, x2)

    def tearDown(self):
        remdb()


class TestGraph(unittest.TestCase):
    def setUp(self):
        initialize()

    def test_graph_dot_gv_file(self):
        def add_yyyy(r):
            r['yyyy'] = r['orderdate'][0:4]
            return r

        def count(rs):
            rs[0]['n'] = len(rs)
            yield rs[0]

        fb.register(
            orders=fb.load('orders.csv', fn=add_yyyy),
            customers=fb.load('customers.csv'),
            # append customer's nationality
            orders1=fb.join(
                ['orders', '*', 'customerid'],
                ['customers', 'Country', 'customerid'],
            ),
            # yearly number of orders by country
            orders2=fb.map(count, 'orders1', by='yyyy, Country'),
        )
        saved_jobs = fb1._JOBS
        fb.run()
        with open('test.gv') as f:
            graph = f.read()
            for j in saved_jobs:
                self.assertTrue(j in graph)

        with fb1._connect('test.db') as c:
            c.drop('customers')

        fb1._JOBS = saved_jobs
        # rerun
        jobs_to_do, jobs_undone = fb.run()
        self.assertEqual(set(j['output'] for j in jobs_to_do),
                         set(['customers', 'orders1', 'orders2']))
        self.assertEqual(jobs_undone, [])

    def tearDown(self):
        remdb()


# Test Join
class TestIntegratedProcess(unittest.TestCase):
    def setUp(self):
        initialize()

    def test_semiannual_and_quartery_orders_average_by_month(self):
        def sumup(rs):
            r = rs[0]
            r['norders'] = len(rs)
            return r

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

        fb.register(
            orders=fb.load('orders.csv'),
            # add month
            orders1=fb.map(add(yyyymm=lambda r: r['orderdate'][:7]), 'orders'),
            # count the number of orders by month
            orders2=fb.map(fn=sumup, data='orders1', by='yyyymm'),
            orders3=fb.map(fn=cnt, data='orders2', by='*'),
            # want to compute past 6 months
            orders4=fb.join(
                ['orders3', '*', 'cnt'],
                ['orders3', 'norders as norders1', 'cnt + 1'],
                ['orders3', 'norders as norders2', 'cnt + 2'],
                ['orders3', 'norders as norders3', 'cnt + 3'],
                ['orders3', 'norders as norders4', 'cnt + 4'],
                ['orders3', 'norders as norders5', 'cnt + 5']
            ),

            orders_avg_nmonth=fb.map(fn=orders_avg_nmonth, data='orders4'),
        )

        fb.run()

        with fb1._connect('test.db') as c:
            xs = []
            for r in fet(c, 'orders_avg_nmonth'):
                if isinstance(r['avg'], float) or isinstance(r['avg'], int):
                    xs.append(xs)
            self.assertEqual(len(xs), 9)

    def tearDown(self):
        remdb()


class TestParallel(unittest.TestCase):
    def setUp(self):
        initialize()

    def test_simple_parallel_work_group(self):
        def add_yyyy(r):
            r['yyyymm'] = r['orderdate'][:7]
            r['yyyy'] = r['orderdate'][:4]
            yield r

        # You can make it slow using expressions like "time.sleep(1)"
        # And see what happens in '_temp' folder
        def count(rs):
            rs[0]['n'] = len(rs)
            yield rs[0]

        def count1(rs):
            rs1 = [r for r in rs if r['yyyy'] == 1997]
            if rs1:
                rs1[0]['n'] = len(rs1)
                yield rs1[0]

        fb.register(
            orders=fb.load('orders.csv', fn=add_yyyy),
            # you can enforce single-core-proc by passing parallel "False"
            orders1=fb.map(count, 'orders', by='yyyymm, shipperid'),
            orders1s=fb.map(count, 'orders', by='yyyymm, shipperid', parallel=True),
            # part of workers do not have work to do, sort of a corner case
            orders2=fb.map(count1, 'orders', by='yyyymm, shipperid'),
            orders2s=fb.map(count1, 'orders', by='yyyymm, shipperid', parallel=True),
            # one column should work as well
            orders3=fb.map(count, 'orders', by='yyyymm'),
            orders3s=fb.map(count, 'orders', by='yyyymm', parallel=True),
        )

        fb.run()
        with fb1._connect('test.db') as c:
            self.assertEqual(list(fet(c, 'orders1')), list(fet(c, 'orders1s')))
            self.assertEqual(list(fet(c, 'orders2')), list(fet(c, 'orders2s')))
            self.assertEqual(list(fet(c, 'orders3')), list(fet(c, 'orders3s')))

    def test_simple_parallel_work_non_group(self):
        def first_name(r):
            r['first_name'] = r['CustomerName'].split()[0]
            yield r

        def first_name1(r):
            if isinstance(r['PostalCode'], int):
                r['first_name'] = r['CustomerName'].split()[0]
                yield r

        fb.register(
            customers=fb.load('customers.csv'),
            customers1=fb.map(first_name, 'customers'),
            customers1s=fb.map(first_name, 'customers', parallel=4),

            customers2=fb.map(first_name1, 'customers'),
            customers2s=fb.map(first_name1, 'customers', parallel=3),

        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(list(fet(c, 'customers1')), list(fet(c, 'customers1s')))
            self.assertEqual(list(fet(c, 'customers2')), list(fet(c, 'customers2s')))

    def test_pmap_with_get(self):
        def orders1():
            d = {}
            for r in fb.read('customers'):
                d[r['CustomerID']] = r['CustomerName']

            def _f(r):
                r['customer_name'] = d.get(r['customerid'], '')
                return r

            return _f

        fb.register(
            customers=fb.load('customers.csv'),
            orders=fb.load('orders.csv'),
            orders1=fb.map(orders1, 'orders'),
            orders2=fb.map(orders1, 'orders', parallel=True),
        )

        fb.run()

        with fb1._connect('test.db') as c:
            names1 = [r['customer_name'] for r in fet(c, 'orders1') if r['customer_name']]
            names2 = [r['customer_name'] for r in fet(c, 'orders2') if r['customer_name']]
            self.assertEqual(len(names1), len(fet(c, 'orders')))
            self.assertEqual(names1, names2)

    def tearDown(self):
        remdb()


class TestLogMsg(unittest.TestCase):
    def test_mute_log_messages(self):
        remdb()
        fb.register(
            orders=fb.load('orders.csv'),
            orders1=fb.map(lambda r: r, 'orders'),
        )
        self.assertEqual(fb1._CONFIG['msg'], True)
        # you can pass keyword args for configuration
        fb.run(msg=False)
        self.assertEqual(fb1._CONFIG['msg'], True)
        remdb()


class TestRun(unittest.TestCase):
    def test_refresh(self):
        fb.register(
            orders=fb.load('orders.csv'),
            products=fb.load('products.csv'),
        )
        fb.run()

        fb1._JOBS = {}
        fb.register(
            orders=fb.load('products.csv')
        )
        fb.run(refresh='orders')
        with fb1._connect('test.db') as c:
            self.assertEqual(len(fet(c, 'orders')), len(fet(c, 'products')))

    def test_export(self):
        initialize()
        if os.path.exists('orders_sample.csv'):
            os.remove('orders_sample.csv')

        fb.register(
            orders_sample=fb.load('orders.csv'),
        )
        fb.run(export='orders_sample')

        fb.register(
            foo=fb.load('orders_sample.csv'),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(fet(c, 'orders_sample'), fet(c, 'foo'))
        if os.path.exists('orders_sample.csv'):
            os.remove('orders_sample.csv')

    # You can export as xlsx, it has advantages when you are dealing with unicode files
    # and also you may find csv insecure sometimes
    def test_export_xlsx(self):
        initialize()
        if os.path.exists('orders_sample.xlsx'):
            os.remove('orders_sample.xlsx')

        fb.register(
            orders_sample=fb.load('orders.csv'),
        )
        fb.run(export='orders_sample.xlsx')
        fb.register(
            foo=fb.load('orders_sample.xlsx'),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(fet(c, 'orders_sample'), fet(c, 'foo'))
        if os.path.exists('orders_sample.xlsx'):
            os.remove('orders_sample.xlsx')


class TestLow(unittest.TestCase):
    def test_low1(self):
        # you can do much more complex joining jobs than this using low
        initialize()

        def join1(seq1, seq2):
            seq1 = groupby(seq1, key=lambda r: r['cust_id'])
            seq2 = groupby(seq2, key=lambda r: r['cust_id'])
            # full join
            for rs1, rs2 in step(seq1, seq2):
                if rs1 and rs2:
                    for r1 in rs1:
                        for r2 in rs2:
                            r1['cust_name'] = r2['cust_name']
                            yield r1

                if not rs1:
                    for r2 in rs2:
                        yield {
                            'order_num': '',
                            'order_date': '',
                            'cust_id': rs2[0]['cust_id'],
                            'cust_name': rs2[0]['cust_name'],
                        }

                if not rs2:
                    for r1 in rs1:
                        r1['cust_name'] = ''
                        yield r1

        fb.register(
            tysql_Orders=fb.load('tysql_Orders.csv'),
            tysql_Customers=fb.load('tysql_Customers.csv'),
            tysql_Orders1=fb.low(join1, [('tysql_Orders', 'cust_id'),
                                         ('tysql_Customers', 'cust_id')]),
        )

        fb.run()
        with fb1._connect('test.db') as c:
            result = []
            for r in fet(c, 'tysql_Orders1'):
                result.append(r['cust_id'])
            self.assertEqual(result, [1000000001, 1000000001, 1000000002, 1000000003, 1000000004, 1000000005])

    def test_low2(self):
        # you can do much more complex joining jobs than this using low
        initialize()

        def join1(seq1, seq2):
            seq1 = groupby(seq1, key=lambda r: r['cust_state'])
            seq2 = groupby(seq2, key=lambda r: r['vend_state'])

            for rs1, rs2 in step(seq1, seq2):

                if rs1 and rs2:
                    for r1 in rs1:
                        for r2 in rs2:
                            r1['vend_name'] = r2['vend_name']
                            yield r1

                if not rs1:
                    # do nothing
                    pass

                if not rs2:
                    for r1 in rs1:
                        r1['vend_name'] = ''
                        yield r1

        fb.register(
            tysql_Customers=fb.load('tysql_Customers.csv'),
            tysql_Vendors=fb.load('tysql_Vendors.csv'),
            tysql_Customers1=fb.low(join1, [('tysql_Customers', 'cust_state'),
                                             ('tysql_Vendors', 'vend_state')]),

        )

        fb.run()

        with fb1._connect('test.db') as c:
            vendors = []
            for r in fet(c, 'tysql_Customers1'):
                vendors.append(r['vend_name'])
            self.assertEqual(vendors, ['', '', '', 'Bears R Us', 'Bear Emporium'])

    # no order cols
    def test_low3(self):
        def firstN(n):
            def fn(seq):
                for _ in range(n):
                    yield next(seq)
            return fn

        initialize()
        fb.register(
            products=fb.load('products.csv'),
            first3=fb.low(firstN(3), [('products', '')]),
            first5=fb.low(firstN(5), [('products', None)]),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(len(fet(c, 'first3')), 3)
            self.assertEqual(len(fet(c, 'first5')), 5)

    def test_low4(self):
        initialize()
        fb.register(
            products=fb.load('products.csv'),
            sample=fb.low(head(5), [('products', None)]),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(len(fet(c, 'sample')), 5)

    def test_low5(self):
        "You don't have to pass tables as args"
        def sample():
            for i in range(5):
                yield {'a': i}
            
        initialize()
        fb.register(
            sample=fb.low(sample),
        )
        fb.run()

        with fb1._connect('test.db') as c:
            self.assertEqual(len(fet(c, 'sample')), 5)


# read & write not tested

class TestException(unittest.TestCase):
    pass


# utils
def nlines_file(name):
    with open(name) as f:
        return len(f.readlines())


def add(**kwargs):
    def fn(r):
        for k, v in kwargs.items():
            r[k] = v(r)
        return r
    return fn


def fet(c, tname):
    return list(c.fetch(f"select * from {tname}"))


if __name__ == "__main__":
    unittest.main()