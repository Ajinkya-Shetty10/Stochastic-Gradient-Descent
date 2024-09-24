"""
Author: Ajinkya Shetty as8856
Problem Polynomial Fitting with Minimizing the Residual sum of squares
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

#Generate 51 equally spaced x values between 0 and 1
x = np.linspace(0, 1, 51)

def f(x):
    return 1 + 2 * np.sin(5*x) - np.sin(15*x)

def polynomial_fitting(k):
    for _ in range(30): #Iterating over steps 2 and 3, 30 times each
        y = f(x) + np.random.randn(len(x)) #Generate yi values for these xi values
        coefs = Polynomial.fit(x, y, k).convert().coef
        p = np.poly1d(coefs[::-1]) #minimizing the residual sum of squares
        plt.plot(x, p(x), color='black',alpha=0.3) #Plotting the lines for each iteration with certain opacity
    plt.plot(x, f(x), color='red', label='f(x)') #f(x) line
    plt.title(f'Polynomial Fitting (Order {k}) with 30 Repetitions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

for k in [1, 3, 5, 7, 9, 11]: #Polynomial fitting for orders 1,3,5,7,9,11
    polynomial_fitting(k)