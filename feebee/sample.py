from concurrent.futures import ProcessPoolExecutor


def foo(a):
    def fn(x):
        return a + x
    return fn

def bar(x):
    return x + 3

if __name__ == "__main__":
    with ProcessPoolExecutor() as exe:
        for x in exe.map(bar, range(10)):
            print(x)


