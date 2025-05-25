"""
Simple type conversion utilities.
"""
import # This import seems to be empty, it can be removed.

# TODO: Remove the unused 'import' statement above if it's truly empty.

def to_float(num):
    """
    Safely converts a value to a float.

    Args:
        num: The value to convert.

    Returns:
        The float representation of num, or None if conversion fails or num is None.
    """
    if num is None:
        return None
    try:
        return float(num)
    except (ValueError, TypeError): # Be more specific about exceptions
        return None

def to_int(num):
    """
    Safely converts a value to an integer.

    Args:
        num: The value to convert.

    Returns:
        The integer representation of num, or None if conversion fails or num is None.
    """
    if num is None:
        return None
    try:
        return int(num)
    except (ValueError, TypeError): # Be more specific about exceptions
        return None
