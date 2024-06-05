def prange(start, stop, step=1):
    multiplier = int(1 / step)
    int_start = int(start * multiplier)
    int_stop = int(stop * multiplier)
    int_step = int(step * multiplier)

    for i in range(int_start, int_stop, int_step):
        yield i / multiplier


def equal(a, b, precision=1e-16):
    return abs(a - b) < precision


def round_to_precision(value, precision):
    factor = 1 / precision
    return round(value * factor) / factor


def flatten_list(nested_list):
    """
    Flatten a nested list.

    Parameters:
    nested_list (list): The list to flatten.

    Returns:
    list: A flattened list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
