import numpy as np
import matplotlib.pyplot as plt

# Set numpy print options to suppress scientific notation
np.set_printoptions(suppress=True)

# Data: Months (x) and Bag Production (y)
months = np.arange(1, 145)
bag_production = np.array([
    1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399,
    2048, 2523, 2086, 2391, 2150, 2340, 3129, 2277, 2964, 2997, 2747, 2862,
    3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439, 3601, 3531,
    3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422,
    4197, 4441, 4736, 4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211,
    4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260, 6110, 5334, 5988, 6235,
    6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594,
    7092, 7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933,
    8756, 8613, 8705, 9098, 8769, 9544, 9050, 9186, 10012, 9685, 9966, 10048,
    10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849, 12123,
    12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162,
    13644, 13808, 14101, 13992, 15191, 15018, 14917, 15046, 15556, 15893,
    16388, 16782, 16716, 17033, 16896, 17689
])

# Plot polynomial interpolation of different orders
plt.figure(figsize=(15, 10))
for order in range(1, 5):
    plt.subplot(2, 2, order)
    plt.plot(months, bag_production, 'r.')
    polynomial_coefficients = np.polyfit(months, bag_production, order)
    polynomial_coefficients = np.round(polynomial_coefficients, 3)  # rounding for clarity
    plt.plot(months, np.polyval(polynomial_coefficients, months), 'b')
    plt.legend(['Bag Production', 'Trend'])
    plt.title(f"Polynomial Interpolation Order {order}")

# Third-order polynomial fit and plot
third_order_coefficients = np.polyfit(months, bag_production, 3)
third_order_coefficients = np.round(third_order_coefficients, 3)

plt.figure(figsize=(12, 5))
plt.plot(months, bag_production, 'r.')
plt.plot(months, np.polyval(third_order_coefficients, months), 'b')
plt.xlabel('Months')
plt.ylabel('Bag Production')
plt.legend(['Bag Production', 'Trend'])
plt.title("EGIER Bag Production Trend (2018-2023) with 3rd Order Polynomial")
plt.show()

# Display polynomial coefficients and equation
print()
print(f'Polynomial coefficients: {third_order_coefficients}')
poly_eq = f'{third_order_coefficients[0]}x^3 + {third_order_coefficients[1]}x^2 + {third_order_coefficients[2]}x + {third_order_coefficients[3]}'
print(f'Polynomial equation: P(x) = {poly_eq}')

# Taylor series components
coefficients = np.array([0.004, -0.134, 47.224, 1748.507])

def polynomial(x):
    return coefficients[0] * x**3 + coefficients[1] * x**2 + coefficients[2] * x + coefficients[3]

def first_derivative(x):
    return 3 * coefficients[0] * x**2 + 2 * coefficients[1] * x + coefficients[2]

def second_derivative(x):
    return 6 * coefficients[0] * x + 2 * coefficients[1]

def third_derivative(x):
    return 6 * coefficients[0]

def taylor_series(x, a):
    return (polynomial(a) +
            first_derivative(a) * (x-a) +
            second_derivative(a) * (x-a)**2 / np.math.factorial(2) +
            third_derivative(a) * (x-a)**3 / np.math.factorial(3))

# Plotting polynomial and Taylor series approximation
plt.figure(figsize=(10, 6))
plt.plot(months, np.polyval(third_order_coefficients, months), 'r', label='Polynomial')
plt.plot(months, [taylor_series(xi, 0) for xi in months], 'b.', label='Taylor Series')
plt.xlabel('Months')
plt.ylabel('Bag Production')
plt.legend()
plt.title('Bag Production Trend: Polynomial vs Taylor Series')
plt.show()

# Display Taylor series expansion around x = 0
a = 0
print()
print(f'Taylor series expansion around x = {a}:')
print(f'{polynomial(a)} + {first_derivative(a)}(x - {a}) + '
      f'({second_derivative(a)}/2!)(x - {a})^2 + '
      f'({third_derivative(a)}/3!)(x - {a})^3')

# Simplified Taylor series expansion
print()
print(f'Simplified Taylor series equation:')
print(f'T(x) = {polynomial(a)} + ({first_derivative(a)})x + '
      f'({second_derivative(a)/2})x^2 + ({third_derivative(a)/6})x^3')

# Bisection method to find root
def find_root(func, low, high, tolerance, max_iter):
    iteration = 1
    while iteration <= max_iter:
        midpoint = (low + high) / 2
        if np.abs(func(midpoint)) < tolerance:
            print(f'Root found: {midpoint} after {iteration} iterations')
            return midpoint
        elif np.sign(func(midpoint)) == np.sign(func(low)):
            low = midpoint
        else:
            high = midpoint
        iteration += 1
    print('Root not found within max iterations')
    return None

def func(x):
    return taylor_series(x, 0) - 25000

root = find_root(func, 1, 200, 0.001, 100)
if root:
    print(f'Max bag storage may be reached at month {round(root)}.')
    print(f'Start building new warehouse by month {round(root) - 13}.')
