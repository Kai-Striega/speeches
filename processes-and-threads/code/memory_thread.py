from time import sleep
from multiprocessing.pool import ThreadPool
XS = [x for x in range(1_000_000)]

def dummy_work(x):
    sleep(10)
    return x

if __name__ == "__main__":
    with ThreadPool() as pool:
        pool.map(dummy_work, [XS, XS, XS])