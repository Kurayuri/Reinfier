def prange(start, stop, step = 1):
    multiplier = int(1 / step)
    int_start = int(start * multiplier)
    int_stop = int(stop * multiplier)
    int_step = int(step * multiplier)
    
    for i in range(int_start, int_stop, int_step):
        yield i / multiplier


def equal(a, b , precision = 1e-16):
    return abs(a - b) < precision