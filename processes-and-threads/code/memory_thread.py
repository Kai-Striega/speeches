from time import sleep
from concurrent.futures import ThreadPoolExecutor
XS = [x for x in range(1_000_000)]

def dummy_work(x):
    sleep(3)
    return x

if __name__ == "__main__":
    with ThreadPoolExecutor(3) as pool:
        pool.map(dummy_work, [XS, XS, XS])
