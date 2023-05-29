import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"⏰⏰ Task '{func.__name__}' started.")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"✅⏰ Task '{func.__name__}' completed in {execution_time:.4f} seconds.")
        return result
    return wrapper
