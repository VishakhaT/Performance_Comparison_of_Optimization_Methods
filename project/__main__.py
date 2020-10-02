# Final Project: Performance optimization over Rosenbrock function
# CSCE 689: Software Engineering
# Texas A&M University
# Name: Vishakha Thakurdwarkar
# An Aggie does not lie, cheat or steal or tolerate those who do.

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import matplotlib.colors as colors
import time

from project.optimize_by_NelderMead import optimize_by_NelderMead
from project.optimize_by_ConjugateGradient import optimize_by_ConjugateGradient
from project.optimize_by_LBFGSB import optimize_by_LBFGSB
from project.optimize_by_NelderMead import generate_rosenbrock_value

def record(xk):
    """
    Store the vector obtained in each iteration in the global variable accessible to all methods
    Input parameter: xk - current parameter vector
    """
    global starting_point
    starting_point = [(-3, -4)]
    starting_point.append(xk)
    return starting_point

def find_minmax(x_iterates, y_iterates):
    """
    Find minimum and maximum in the x and y vectors
    """
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    
    for i in range(len(x_iterates)):
        if x_iterates[i] < x_min:
            x_min = x_iterates[i]

    for i in range(len(x_iterates)):
        if x_iterates[i] > x_max:
            x_max = x_iterates[i]

    for i in range(len(y_iterates)):
        if y_iterates[i] < y_min:
            y_min = y_iterates[i]

    for i in range(len(y_iterates)):
        if y_iterates[i] > y_max:
            y_max = y_iterates[i]
            
    return x_min, x_max, y_min, y_max


def main():
    """
    Main entry point for application
    """
    
    #start the time counter
    start_time = time.perf_counter()
    print("Optimization Research")
    
    #Method 1 - Nelder-Mead
    x_iterates, y_iterates, NM_time, NM_index, starting_point = optimize_by_NelderMead()
    #plot of Optimization using Nelder-Mead method
    plt.plot(x_iterates, y_iterates, 'ro:', linewidth=3, label="Nelder-Mead")    
    #finding extremeties
    x_min, x_max, y_min, y_max = find_minmax(x_iterates, y_iterates)
    
    #Method 2 - Conjugate Gradient 
    x_iterates, y_iterates, CG_time, CG_index, starting_point = optimize_by_ConjugateGradient()   
    #plot of Optimization using Conjugate Gradient method
    plt.plot(x_iterates, y_iterates, 'mo--', linewidth=3, label="Conjugate Gradient")    
    #finding extremeties
    x_min, x_max, y_min, y_max = find_minmax(x_iterates, y_iterates)

    #Method 3 - L-BFGS-B
    x_iterates, y_iterates, LB_time, LB_index, starting_point = optimize_by_LBFGSB()
    #plot of Optimization using L-BFGS-B method
    plt.plot(x_iterates, y_iterates, 'co-', linewidth=3, label="L-BFGS-B")
    #finding extremeties
    x_min, x_max, y_min, y_max = find_minmax(x_iterates, y_iterates)
    
    #Printing comparison of three optimization methods
    print("| Method             | Iterations | Time (ms) |")
    print("+--------------------+------------+-----------+")
    print("| Nelder-Mead        |", '{:10d}'.format(NM_index), "|", '{:9.2f}'.format(1E3*NM_time), "|")
    print("| Conjugate Gradient |", '{:10d}'.format(CG_index), "|", '{:9.2f}'.format(1E3*CG_time), "|")
    print("| L-BFGS-B           |", '{:10d}'.format(LB_index), "|", '{:9.2f}'.format(1E3*LB_time), "|")
    
    num_points = 1000
    x = np.linspace(x_min-1, x_max+1, num_points)
    y = np.linspace(y_min-1, y_max+1, num_points)

    X, Y = np.meshgrid(x, y)
    Z = generate_rosenbrock_value([X,Y])

    end_time = time.perf_counter()
    
    #Total execution time
    print("| Total Execution Time (w/o plot):    ", '{:5.2f}'.format(1E3*(end_time - start_time)), "|")

    normalize = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    plt.contourf(X, Y, Z, norm=normalize, cmap='cividis', locator=LogLocator(base=2, numticks=20))

    #Plotting all the graphs of three methods together
    plt.plot(-3, -4, 'kX', markersize=12, markeredgecolor='white', label='Starting Point')
    plt.plot(1, 1, 'wX', markersize=12, markeredgecolor='black', label='Global Minimum')

    plt.legend(loc=2)
    plt.title('Comparison of Three Optimization Methods\nOptimizing Over the Rosenbrock Function')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    plt.show()


if __name__ == '__main__':
    main()