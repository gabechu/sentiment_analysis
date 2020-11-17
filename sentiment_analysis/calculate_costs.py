"""Cost of running sentiment analysis services"""

from typing import List
import math


def calculate_comprehend_cost_no_batch(texts: List[str]) -> float:
    # character encoding is UTF8 according to
    # https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
    # measured in units of 100 characters
    units = map(lambda text: math.ceil(len(text) / 100), texts)
    # minimum charge per request is 3 unit (300 characters)
    revised_units = map(lambda length: length if length > 3 else 3, units)

    total_cost = total_units = 0
    for curnt_units in list(revised_units):
        total_units += curnt_units
        # Price Per Unit
        # https://aws.amazon.com/comprehend/pricing/
        if total_units <= 10000000:
            total_cost += curnt_units * 0.0001
        elif total_units > 10000000 and total_units <= 50000000:
            total_cost += curnt_units * 0.00005
        elif total_cost > 50000000:
            total_cost += curnt_units * 0.000025

    return total_cost
