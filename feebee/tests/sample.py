import os
import sys

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

import feebee as fb
#
def foo(r):
    r['x'] = r['customerid']
    yield r 


fb.register(
    orders = fb.load('orders.csv'),
    customers = fb.load('customers.csv'),
    orders1 = fb.map(foo, 'orders', parallel=4),
)

if __name__ == "__main__":
    fb.run()