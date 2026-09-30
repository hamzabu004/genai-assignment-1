import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    yield result
    result["ms"] = (time.perf_counter() - start) * 1000
