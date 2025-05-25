import


def to_float(num):
    if num is None:
        return None
    try:
        return float(num)
    except:
        return None
def to_int(num):
    if num is None:
        return None
    try:
        return int(num)
    except:
        return None
