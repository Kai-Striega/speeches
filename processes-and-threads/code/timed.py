from functools import wraps
from time import time


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time() * 1_000
        result = f(*args, **kwargs)
        end = time() * 1_000
        time_ms = end - start
        print(f"{f.__name__}: {time_ms:.2f}ms")
        return result
    return wrapper
