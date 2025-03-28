import matplotlib.pyplot as plt
import numpy as np

# Example data – replace with your actual values
distance = np.array([548,542,534,527,520,513,506,499,491,485,480])
distance = distance/1000
weight = np.array([0.01503, 0.01996, 0.02507, 0.03,0.03523, 0.04016,0.04527,0.0502,0.05543,0.06036,0.06467])
dis_err = np.array([39,53,7.2,7.1,9,4.9,5,6,7.3,5.5,6])
dis_err = dis_err/100000
wght_err = np.array([1, 1.41, 1.41, 1.73, 1.73, 2, 1.73, 2, 2, 2.24, 1.41])
wght_err = wght_err/10000
# Linear fit
coeffs = np.polyfit(weight, distance, 1)  # 1st degree polynomial
slope, intercept = coeffs
fit_line = slope * weight + intercept

# R-squared calculation
y_mean = np.mean(distance)
ss_tot = np.sum((distance - y_mean) ** 2)
ss_res = np.sum((distance - fit_line) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Plotting
plt.errorbar(weight, distance, xerr=wght_err, yerr=dis_err, fmt='o', ecolor='blue', capsize=5, label='Data with errors')
plt.plot(weight, fit_line, 'r-', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}')

# Labels and legend
plt.xlabel('Weight values')
plt.ylabel('Distance values')
plt.title('Scatter Diagram with Error Bars and Linear Fit')
plt.grid(True)
plt.legend()

# Show plot
plt.show()
