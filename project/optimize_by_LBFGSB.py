import numpy as np
import scipy.optimize
import time

def generate_rosenbrock_value(starting_point):
    """
    Return the value of Rosenbrock function for input vectors
    """
    return pow(1 - starting_point[0], 2) + 100 * pow(starting_point[1] - pow(starting_point[0], 2), 2)

def record(xk):
    """
    Store the vector obtained in each iteration in the global variable accessible to all methods
    Input parameter: xk - current parameter vector
    """
    global starting_point
    starting_point.append(xk)

def optimize_by_LBFGSB():
    """
    Optimize the performance over Rosenbrock function using L-BFGS-B method
    """
    #setting the starting point for Rosenbrock equation
    global starting_point
    starting_point = [(-3, -4)]

    #Optimization using L-BFGS-B method
    start_time_LBmethod = time.perf_counter()
    result = scipy.optimize.minimize(generate_rosenbrock_value, starting_point, method='L-BFGS-B', callback=record)
    end_time_LBmethod = time.perf_counter()
    LB_time = end_time_LBmethod - start_time_LBmethod
    LB_index = len(starting_point)

    x_iterates = [xk[0] for xk in starting_point]
    y_iterates = [xk[1] for xk in starting_point]
    
    return x_iterates, y_iterates, LB_time, LB_index, starting_point