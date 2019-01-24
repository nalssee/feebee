from tqdm import tqdm
import time


def foo():
    with tqdm(total=10) as pbar:
        yield 3
        pbar.update(4)
        yield 4
        pbar.update(6)

for i in foo():
    time.sleep(0.1)

