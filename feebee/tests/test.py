import os
import sys
import unittest

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
from feebee.config import _DBNAME, _GRAPH_NAME
from feebee.util import add, isnum, where



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
    if os.path.isfile(_DBNAME):
        os.remove(_DBNAME)


def initialize():
    remdb()
    #
    fb.feebee._JOBS = {}
    fb.run()



class TestLoading(unittest.TestCase):
    def setUp(self):
        initialize()
        # make empty table

    def test_dbfilename_created_as_script_name(self):
        name, _ = os.path.splitext(os.path.basename(__file__))
        self.assertIn(name + '.db', os.listdir())

    def test_loading_ordinary_csv(self):
        fb.register(orders=fb.load('orders.csv'))
        fb.run()
        self.assertEqual(len(list(fb.get('orders'))), fb.feebee._line_count('orders.csv', 'utf-8', '\n'))

    # TODO: some of the other options like encoding must be tested
    def test_loading_ordinary_tsv(self):
        fb.register(markit=fb.load('markit.tsv'))
        fb.run()
        self.assertEqual(len(list(fb.get('markit'))), fb.feebee._line_count('markit.tsv', 'utf-8', '\n'))

    def test_loading_semicolon_separated_file(self):
        fb.register(orders1=fb.load('orders1.txt', delimiter=";"))
        fb.run()

        self.assertEqual(len(list(fb.get('orders1'))), fb.feebee._line_count('orders1.txt', 'utf-8', '\n'))
        with open('orders1.txt') as f:
            cols = list(fb.get('orders1')[0])
            self.assertEqual(f.readline()[:-1].split(";"), cols)

    def test_loading_excel_files(self):
        fb.register(
            ff=fb.load('ff.xls'),
            ff1=fb.load('ff.xlsx'),
            na_sample=fb.load('na_sample.xlsx'),
        )
        fb.run()

        self.assertEqual(fb.get('ff'), fb.get('ff1'))
        na_sample = fb.get('na_sample')
        self.assertEqual(list(na_sample[0]), ['col1', 'col2'])
        self.assertEqual(len(na_sample), 2)

    def test_loading_sas_file(self):
        fb.register(
            ff5=fb.load('ff5_ew_mine.sas7bdat'),
        )
        fb.run()
        ff5 = fb.get('ff5', df=True)
        self.assertEqual(len(ff5), 253)

    def test_loading_stata_file(self):
        fb.register(
            crime=fb.load('crime.dta'),
        )
        fb.run()
        self.assertEqual(len(fb.get('crime', df=True)), 2725)

    def test_loading_seq(self):
        def add3(r):
            r['b'] = r['a'] + 3
            return r

        fb.register(
            seq_loading_sample=fb.load(({'a': i} for i in range(10)), fn=add3)
        )
        fb.run()

        seq_loading_sample = fb.get('seq_loading_sample')
        self.assertEqual(list(seq_loading_sample[0]), ['a', 'b'])

    def tearDown(self):
        remdb()


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
            vendors1=fb.apply(input_none, 'vendors'),
            vendors2=fb.apply(input_none1, 'vendors'),
        )
        fb.run()

        self.assertEqual(len([r for r in fb.get('vendors') if r['vend_state'] == '  ']), 1)
        self.assertEqual(len([r for r in fb.get('vendors') if r['vend_state'] == '']), 1)
        # '' is falsy but '  ' is not
        self.assertEqual(len([r for r in fb.get('vendors1') if r['vend_state'] is None]), 1)
        self.assertEqual(len([r for r in fb.get('vendors2') if r['vend_state'] is None]), 0)

    def tearDown(self):
        remdb()


class TestApply(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            orders=fb.load('orders.csv'),
            customers=fb.load('customers.csv'),
        )
        fb.run()

    def test_thunk(self):
        def count():
            customers = fb.get('customers')
            def _f(r):
                r['num'] = len(customers)
                return r
            return _f

        fb.register(
            orders1 = fb.apply(count, 'orders'),
            orders2 = fb.apply(count, 'orders', parallel=True),
        )
        fb.run()

        orders1 = fb.get('orders1')
        orders2 = fb.get('orders2')
        self.assertEqual([r['num'] for r in orders1], [91] * len(orders1))
        self.assertEqual(orders1, orders2)

    def test_append_yyyy_yyyymm_columns(self):
        def add_yyyy_yyyymm(r):
            if r['orderdate'] > '1996-xx-xx':
                r['yyyy'] = r['orderdate'][:4]
                r['yyyymm'] = r['orderdate'][:7]
                yield r

        fb.register(
            orders1=fb.apply(add_yyyy_yyyymm, 'orders'),
        )
        fb.run()

        orders = fb.get('orders')
        orders1 = fb.get('orders1')
        self.assertEqual(len(orders[0]) + 2, len(orders1[0]))
        self.assertEqual(sum(1 for _ in orders1), 44)

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
            customers1=fb.apply(bigmarket(5), 'customers', by='Country'),
            customers2=fb.apply(bigmarket1(5), 'customers', by='Country')
        )
        fb.run()
        self.assertEqual(list(fb.get('customers1')), list(fb.get('customers2')))

    def test_group_n(self):
        fb.register(
            orders1=fb.apply(lambda rs: {'a': len(rs)}, 'orders', by=10)
        )
        fb.run()
        self.assertEqual(len(fb.get('orders')), sum(r['a'] for r in fb.get('orders1')))

    # all of them at once
    def test_group_star(self):
        fb.register(
            orders1=fb.apply(lambda rs: rs, 'orders', by=' * '),
        )
        fb.run()
        self.assertEqual(fb.get('orders'), fb.get('orders1'))

    def test_group_invalid(self):
        fb.register(
            orders1=fb.apply(lambda rs: {'a': 10}, 'orders', by=10.0),
        )
        _, undone = fb.run()
        self.assertEqual([x['output'] for x in undone], ['orders1'])

    def test_insert_empty_rows(self):
        def filter10(r):
            if r['shipperid'] == 10:
                yield r

        fb.register(
            orders1=fb.apply(filter10, 'orders')
        )
        fb.run()
        with self.assertRaises(fb.exceptions.NoSuchTableFound):
            list(fb.get('orders1'))

    def test_return_none(self):
        def foo(r):
            if r['shipperid'] == 1:
                return r

        fb.register(
            orders1=fb.apply(foo, 'orders'),
        )
        fb.run()
        self.assertEqual(len(list(fb.get('orders1'))), 54)

    def test_get(self):
        # dataframe
        orders_df = fb.get('orders', df=True)
        # list of dicts
        orders_ld = fb.get('orders')
        self.assertEqual(orders_df.shape[0], len(orders_ld))
        self.assertEqual(orders_df.shape[1], len(orders_ld[0]))

    def test_join(self):
        fb.register(
            torders = fb.load('tysql_Orders.csv'),
            torder_items = fb.load('tysql_OrderItems.csv'),
            torders1 = fb.apply(add(cust_id1=lambda r: str(r['cust_id'])[-1]), 'torders'),
            torders2 = fb.join(
                ['torders1', "*", 'order_num, cust_id1'],
                ['torder_items', 'order_num as order_num1, order_item, prod_id, quantity', ['order_num ', (">", 'order_item')]]
            ),
            torders3 = fb.apply(where(lambda r: isnum(r['order_num1'])), 'torders2')

        )
        fb.run()
        self.assertEqual(len(fb.get('torders2')), 11)
        self.assertEqual(len(fb.get('torders3')), 9)

    def tearDown(self):
        remdb()


class TestApplyErrornousInsertion(unittest.TestCase):
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

        fb.register(orders1=fb.apply(errornous1, 'orders', by='*'))
        fb.run()
        # orders1 is not created
        with self.assertRaises(fb.feebee.NoSuchTableFound):
            list(fb.get('orders1'))

    def test_insert_1col_deleted(self):
        def errornous1(rs):
            r = rs[0]
            yield r
            del r['customerid']
            yield r

        fb.register(orders1=fb.apply(errornous1, 'orders', by='*'))
        fb.run()
        # orders1 is not created
        with self.assertRaises(fb.exceptions.NoSuchTableFound):
            list(fb.get('orders1'))

    def test_insert_1col_added(self):
        def errornous1(rs):
            r = rs[0]
            yield r
            # added 'xxx' col is ignored
            r['xxx'] = 10
            yield r

        fb.register(orders1=fb.apply(errornous1, 'orders', by='*'))
        fb.run()
        x1, x2 = list(fb.get('orders1'))
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
            orders2=fb.apply(count, 'orders1', by='yyyy, Country'),
        )
        saved_jobs = fb.feebee._JOBS
        fb.run()
        with open(_GRAPH_NAME) as f:
            graph = f.read()
            for j in saved_jobs:
                self.assertTrue(j in graph)

        with fb.feebee._connect(_DBNAME) as c:
            c.drop('customers')

        fb.feebee._JOBS = saved_jobs
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
            orders1=fb.apply(add(yyyymm=lambda r: r['orderdate'][:7]), 'orders'),
            # count the number of orders by month
            orders2=fb.apply(fn=sumup, data='orders1', by='yyyymm'),
            orders3=fb.apply(fn=cnt, data='orders2', by='*'),
            # want to compute past 6 months
            orders4=fb.join(
                ['orders3', '*', 'cnt'],
                ['orders3', 'norders as norders1', 'cnt + 1'],
                ['orders3', 'norders as norders2', 'cnt + 2'],
                ['orders3', 'norders as norders3', 'cnt + 3'],
                ['orders3', 'norders as norders4', 'cnt + 4'],
                ['orders3', 'norders as norders5', 'cnt + 5']
            ),

            orders_avg_nmonth=fb.apply(fn=orders_avg_nmonth, data='orders4'),
        )

        fb.run()

        # with mo1._connect('test.db') as c:
        xs = []
        for r in fb.get('orders_avg_nmonth'):
            if isinstance(r['avg'], float) or isinstance(r['avg'], int):
                xs.append(xs)
        self.assertEqual(len(xs), 9)

    def tearDown(self):
        remdb()


class Testconcat(unittest.TestCase):
    def setUp(self):
        initialize()

    def test_simple_union(self):
        fb.register(
            orders = fb.load('orders.csv'),
            orders1=fb.apply(lambda r: r, 'orders'),
            orders2=fb.concat('orders, orders1'),
            # the following is also fine
            # orders2=fb.append(['orders', 'orders1'])
        )
        fb.run()

        self.assertEqual(len(list(fb.get('orders'))) * 2, len(list(fb.get('orders2'))))

    def tearDown(self):
        remdb()


class Testmzip(unittest.TestCase):
    def setUp(self):
        initialize()
        fb.register(
            order_items = fb.load('tysql_OrderItems.csv'),
            products = fb.load('tysql_Products.csv'),
        )
        fb.run()

    def test_mzip1(self):
        def itemsfn():
            allproducts = fb.get('products')
            def _f(items, products):
                if items:
                    for item in items:
                        r = {'order_num': item['order_num']}
                        r['n']  = len(allproducts)
                        r['prod_desc'] = ''
                        if products:
                            r['prod_desc'] = products[0]['prod_desc']
                        yield r
                else:
                    yield {'order_num': '', 'n': len(allproducts), 'prod_desc': 'Empty'}

            return _f

        fb.register(
            items1 = fb.mzip(itemsfn, [('order_items', 'prod_id'), ('products', 'prod_id')]),
            items2 = fb.mzip(itemsfn, [('order_items', 'prod_id'), ('products', 'prod_id')], stop_short=True)
        )
        fb.run()

        self.assertEqual(len(fb.get('items1')), len(fb.get('items2')) + 2)

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
            orders1=fb.apply(count, 'orders', by='yyyymm, shipperid'),
            orders1s=fb.apply(count, 'orders', by='yyyymm, shipperid', parallel=True),
            # part of workers do not have work to do, sort of a corner case
            # orders2=fb.apply(count1, 'orders', by='yyyymm, shipperid'),
            # orders2s=fb.apply(count1, 'orders', by='yyyymm, shipperid', parallel=True),
            # # one column should work as well
            # orders3=fb.apply(count, 'orders', by='yyyymm'),
            # orders3s=fb.apply(count, 'orders', by='yyyymm', parallel=True),

        )

        fb.run()
        # self.assertEqual(list(fb.get('orders1')), list(fb.get('orders1s')))
        # self.assertEqual(list(fb.get('orders2')), list(fb.get('orders2s')))
        # self.assertEqual(list(fb.get('orders3')), list(fb.get('orders3s')))

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
            customers1=fb.apply(first_name, 'customers'),
            customers1s=fb.apply(first_name, 'customers', parallel=4),

            customers2=fb.apply(first_name1, 'customers'),
            customers2s=fb.apply(first_name1, 'customers', parallel=3),

        )
        fb.run()
        print(len(fb.get('customers1')))
        print(len(fb.get('customers1s')))

        # self.assertEqual(list(fb.get('customers1')), list(fb.get('customers1s')))
        # self.assertEqual(list(fb.get('customers2')), list(fb.get('customers2s')))

    def test_parallel_apply_with_get(self):
        def orders1():
            d = {}
            for r in fb.get('customers'):
                d[r['CustomerID']] = r['CustomerName']
            def _f(r):
                r['customer_name'] = d.get(r['customerid'], '')
                return r
            return _f

        fb.register(
            customers=fb.load('customers.csv'),
            orders=fb.load('orders.csv'),
            orders1=fb.apply(orders1, 'orders'),
            orders2=fb.apply(orders1, 'orders', parallel=True),
        )

        fb.run()

        names1 = [r['customer_name'] for r in fb.get('orders1') if r['customer_name']]
        names2 = [r['customer_name'] for r in fb.get('orders2') if r['customer_name']]
        self.assertEqual(len(names1), len(fb.get('orders')))
        self.assertEqual(names1, names2)

    def tearDown(self):
        remdb()


class TestLogMsg(unittest.TestCase):
    def test_mute_log_messages(self):
        remdb()
        fb.register(
            orders=fb.load('orders.csv'),
            orders1=fb.apply(lambda r: r, 'orders'),
        )
        self.assertEqual(fb.config._CONFIG['msg'], True)
        # you can pass keyword args for configuration
        fb.run(msg=False)
        self.assertEqual(fb.config._CONFIG['msg'], True)
        remdb()


class TestRun(unittest.TestCase):
    def test_refresh(self):
        fb.register(
            orders=fb.load('orders.csv'),
            products=fb.load('products.csv'),
        )
        fb.run()

        fb.feebee._JOBS = {}
        fb.register(
            orders=fb.load('products.csv')
        )
        fb.run(refresh='orders')
        self.assertEqual(len(fb.get('orders')), len(fb.get('products')))

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

        self.assertEqual(fb.get('orders_sample'), fb.get('foo'))

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

        self.assertEqual(fb.get('orders_sample'), fb.get('foo'))
        if os.path.exists('orders_sample.xlsx'):
            os.remove('orders_sample.xlsx')

# read & write not tested

class TestException(unittest.TestCase):
    pass


# utils
# def nlines_file(name):
#     with open(name) as f:
#         return len(f.readlines())


if __name__ == "__main__":
    unittest.main()
