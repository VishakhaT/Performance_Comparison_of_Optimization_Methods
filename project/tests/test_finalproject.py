import pytest
import re
import numpy as np

from project.optimize_by_NelderMead import generate_rosenbrock_value
from project.__main__ import record
from project.__main__ import find_minmax

def test_rosenbrock_value():
    """
    Test the function generate_rosenbrock_value(starting_point)

    Checks to see if the value calculated by Rosenbrock function is correct
    """
    test_array = [1,3]
    value = generate_rosenbrock_value(x)
    assert(value == 400)
    
def test_record():
    """
    Test the function record(xk)

    Checks to see if the vector is stored after each iteration
    """
    test_array = [1, 3]
    output = record(x)
    assert(len(output) == 2)
    assert(output[0] == (-3, -4))
    assert(output[1] == [1, 3])
    
def test_find_minmax():
    """
    Test the function find_minmax(x_iterates, y_iterates)
    
    Checks to see if the minimum and maximum calculated by the function are correct
    """
    test_array1 = [0, 2, 3, 4]
    test_array2 = [2, -4, 1, 0]
    x_min, x_max, y_min, y_max = find_minmax(test_array1, test_array2)
    assert(x_min == 0)
    assert(x_max == 4)
    assert(y_min == -4)
    assert(y_max == 2)
    
    
