import numpy as np
import matplotlib.pyplot as plt


def linear_approximation(x, y):
    # Find the coefficients for linear function using least squares method
    a, b = np.polyfit(x, y, 1)

    # Round the coefficients to 0.01
    a = round(a, 2)
    b = round(b, 2)

    # Define the linear function
    def f(x):
        return a * x + b

    return f


def power_approximation(x, y):
    x_log = np.log(x)
    y_log = np.log(y)
    A = np.vstack([x_log, np.ones(len(x_log))]).T
    a, b = np.linalg.lstsq(A, y_log, rcond=None)[0]
    return lambda x: np.exp(a * np.log(x) + b)



def exponential_approximation(x, y):
    # Find the coefficients for exponential function using least squares method
    coefficients = np.polyfit(x, np.log(y), 1)
    a = np.exp(coefficients[1])
    b = coefficients[0]

    # Round the coefficients to 0.01
    a = round(a, 2)
    b = round(b, 2)

    # Define the exponential function
    def f(x):
        return a * np.exp(b * x)

    return f


def quadratic_approximation(x, y):
    # Find the coefficients for quadratic function using least squares method
    a, b, c = np.polyfit(x, y, 2)

    # Round the coefficients to 0.01
    a = round(a, 2)
    b = round(b, 2)
    c = round(c, 2)

    # Define the quadratic function
    def f(x):
        return a * x ** 2 + b * x + c

    return f

# Test the functions on given points
x = np.array([3, 5, 7, 9, 11, 13])
y = np.array([3.5, 4.4, 5.7, 6.1, 6.5, 7.3])

linear_func = linear_approximation(x, y)
power_func = power_approximation(x, y)
exp_func = exponential_approximation(x, y)
quad_func = quadratic_approximation(x, y)

# Plot the graphs of the obtained functions
xx = np.linspace(x.min(), x.max(), 100)

plt.plot(xx, linear_func(xx), label='Linear')
plt.plot(xx, power_func(xx), label='Power')
plt.plot(xx, exp_func(xx), label='Exponential')
plt.plot(xx, quad_func(xx), label='Quadratic')

# Plot the given points
plt.scatter(x, y, label='Given Points')

plt.legend()
plt.show()





