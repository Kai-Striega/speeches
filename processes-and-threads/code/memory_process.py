from time import sleep
from multiprocessing import Pool
XS = [x for x in range(1_000_000)]

def dummy_works(x):
    sleep(10)
    return x

if __name__ == "__main__":
    with Pool() as pool:
        pool.map(dummy_works, [XS, XS, XS])
