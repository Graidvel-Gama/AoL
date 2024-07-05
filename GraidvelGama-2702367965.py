import numpy as np
import matplotlib.pyplot as plt

y = [1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399, 2048, 2523, 2086, 2391, 2150, 2340, 3129, 2277, 2964, 2997, 2747, 2862, 3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439, 3601, 3531, 3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422, 4197, 4441, 4736, 4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211, 4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260, 6110, 5334, 5988, 6235, 6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594, 7092, 7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933, 8756, 8613, 8705, 9098, 8769, 9544, 9050, 9186, 10012, 9685, 9966, 10048, 10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849, 12123, 12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162, 13644, 13808, 14101, 13992, 15191, 15018, 14917, 15046, 15556, 15893, 16388, 16782, 16716, 17033, 16896, 17689] # Corresponding data
x = list(range(1, len(y) + 1)) # Time stamp, in this case per month

# Taylor Expansion Function
def taylor_expansion(x, a, b, n_terms):
    # Create an array to fit all values, in this case as much as x_div's value
    approx = np.zeros_like(x, dtype=float)

    # Repeat the loop n times
    for n in range(n_terms):
        # In this calculation, we are using Maclaurin Equation, so with x = 0, we get:
        # a * (1 + bx + ((bx)^2)/2! + ((bx)^3)/3! + ...) 
        approx += (b*x)**n / np.math.factorial(n)
    return a * approx

import numpy as np
import matplotlib.pyplot as plt

# Taylor Expansion Function
def taylor_expansion(x, a, b, n_terms):
    # Create an array to fit all values, in this case as much as x_div's value
    approx = np.zeros_like(x, dtype=float)

    # Repeat the loop n times
    for n in range(n_terms):
        # In this calculation, we are using Maclaurin Equation, so with x = 0, we get:
        # a * (1 + bx + ((bx)^2)/2! + ((bx)^3)/3! + ...) 
        approx += (b * x)**n / np.math.factorial(n)
    return a * approx

# Actual data
y = [1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399, 2048, 2523, 2086, 2391, 2150, 2340, 3129, 2277, 2964, 2997, 2747, 2862, 3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439, 3601, 3531, 3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422, 4197, 4441, 4736, 4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211, 4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260, 6110, 5334, 5988, 6235, 6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594, 7092, 7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933, 8756, 8613, 8705, 9098, 8769, 9544, 9050, 9186, 10012, 9685, 9966, 10048, 10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849, 12123, 12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162, 13644, 13808, 14101, 13992, 15191, 15018, 14917, 15046, 15556, 15893, 16388, 16782, 16716, 17033, 16896, 17689]
x = list(range(1, len(y) + 1)) # Time stamp, in this case per month

x = np.array(x)
y = np.array(y)

# Since this script is supported with Exponential Equation (y = a + e^(bx)), we need to find a and b first.
# The idea is to take the natural logarithm of both sides, so log(y) = log(a) + bx.
# This creates a linear equation with both logarithmic sides. If log(y) = Y and log(a) = A, then Y = bx + A,
# So...

# Take natural logarithm of y.
y_log = np.log(y)

# Perform linear regression to estimate parameters a_log and b_log.
# This is not linear approach. This linear regression is simply to find 2 missing variables.
A = np.vstack([x, np.ones_like(x)]).T

# Since we are using y_log, a and b will also in the natural logarithm form. hence...
b_log, a_log = np.linalg.lstsq(A, y_log, rcond=None)[0]

# Calculate parameters a and b for the exponential model
# Since A = log(a), returning it means exponentializing it, so...
a = np.exp(a_log)

# And since b = b, b_log can be directly used to b.
b = b_log

# To achieve the curve needed, divide x to many divisions, in this case 100 times across 1 to the length of y.
x_div = np.linspace(1, len(y), 100)

# Substitute x_div, a, b, and total regression loop to the function taylor_expansion.
new_y = taylor_expansion(x_div, a, b, 10)

target_production = 25000

# This prints out the Exponential Equation that is used for this data
print("Current equation used:")
print(f"y = {a} * e^({b}x)")

# This prints all the data for each month.
for n in range(len(new_y)):
    print(new_y[n])

x_target = (np.log(target_production / a)) / b
print(f"The bag production is expected to reach {target_production} after approximately {x_target:.2f} months.")

max_div = np.linspace(1, x_target, 100)
max_y = taylor_expansion(max_div, a, b, 10)

# Draw the graph, as usual
plt.figure(figsize=(10, 6))
plt.plot(x, y, '.', label='Original data',)
plt.plot(max_div, max_y, 'b--', label='Max Taylor Approximation')
plt.plot(x_div, new_y, 'r-', label='Taylor Approximation')
plt.xlabel('x (Data Period)')
plt.ylabel('y (Bag Production)')
plt.title('Taylor Series Approximation and Exponential Fit Equation to given data')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
