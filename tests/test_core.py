import os
import shutil
import pytest
import feebee as fb

# much more work to do

@pytest.fixture(autouse=True)
def run_around_tests():
    files = ['pytest.db', 
             'pytest.gv', 
             'pytest.gv.pdf']
    def remfiles():
        for fname in files:
            if os.path.isfile(fname):
                os.remove(fname)
        fb.feebee._JOBS = {}
        shutil.rmtree(fb.feebee._TEMP, ignore_errors=True)
    remfiles()
    yield
    remfiles()


def test_loading_ordinary_csv():
    fb.register(orders=fb.load('tests/Orders.csv'))
    fb.run()
    orders1 = fb.get('orders')
    orders2 = fb.util.readxl('tests/Orders.csv')
    header = next(orders2)
    assert list(orders1[0].keys()) == header 

    for a, b in zip(orders1, orders2):
        assert list(a.values()) == b


def test_apply_order_year():
    def year(r):
        r['order_year'] = r['order_date'][:4]
        return r

    fb.register(
        orders=fb.load('tests/Orders.csv'),
        orders1=fb.apply(year, 'orders')
    )

    fb.run()
    for r in fb.get('orders1'):
        assert r['order_year'] == int(r['order_date'][:4])


def test_apply_group1():
    def size(rs):
        r0 = rs[0]
        r0['n'] = len(rs)
        return r0

    fb.register(
        order_items=fb.load('tests/OrderItems.csv'),
        order_items1=fb.apply(size, 'order_items', by='prod_id'),
        order_items2=fb.apply(size, 'order_items', by='prod_id, order_item'),
    )
    fb.run()
    assert len(fb.get('order_items1')) == 7
    assert len(fb.get('order_items2')) == 16 


def test_join():
    fb.register(
        products = fb.load('tests/Products.csv'),
        vendors = fb.load('tests/Vendors.csv'),
        products1 = fb.join(
            ['products', '*', 'vend_id'],
            ['vendors', 'vend_name, vend_country', 'vend_id'],
        )
    )
    fb.run()
    products1 = fb.get('products1')
    assert products1[0]['vend_country'] == 'USA'
    assert products1[-1]['vend_country'] == 'England'


def test_parallel1():
    def revenue(r):
        r['rev'] = r['quantity'] * r['item_price']
        return r

    fb.register(
        items = fb.load('tests/OrderItems.csv'),
        items1 = fb.apply(revenue, 'items'),
        items2 = fb.apply(revenue, 'items', parallel=True),

    )
    fb.run()
    assert fb.get('items1') == fb.get('items2')


def test_parallel2():
    def size(rs):
        r0 = rs[0]
        r0['n'] = len(rs)
        return r0

    fb.register(
        items = fb.load('tests/OrderItems.csv'),
        items1 = fb.apply(size, 'items', by='prod_id'),
        items2 = fb.apply(size, 'items', by='prod_id', parallel=True),

    )
    fb.run()
    assert fb.get('items1') == fb.get('items2')

