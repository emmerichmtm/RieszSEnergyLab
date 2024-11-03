import numpy as np
import matplotlib.pyplot as plt

# Define fixed points x1, x2, x4
x1 = 0.0
x2 = 1.0
x4 = 3.0
s = 2.0  # Set Riesz s-energy parameter

# Define the range for x3 between 1.5 and 2.5
x3_values = np.linspace(1.5, 2.5, 100)

# Compute the Riesz s-energy for each x3
riesz_energy = (1 / np.abs(x3_values - x1)**s +
                1 / np.abs(x3_values - x2)**s +
                1 / np.abs(x4 - x3_values)**s +
                1 / np.abs(x4 - x2)**s +
                1 / np.abs(x4 - x1)**s +
                1 / np.abs(x2 - x1)**s)

# Find the minimizer of the Riesz s-energy (the x3 value that minimizes the energy)
minimizer_x3 = x3_values[np.argmin(riesz_energy)]
minimizer_log_energy = np.log(riesz_energy[np.argmin(riesz_energy)])

# Plot the log of the Riesz s-energy with a line at the minimizer
plt.figure(figsize=(8, 6))
plt.plot(x3_values, np.log(riesz_energy), label="Log Riesz s-energy for x3")
plt.axvline(x=minimizer_x3, color='red', linestyle='--', label=f"Minimizer at x3 = {minimizer_x3:.2f}")
plt.xlabel("$x_3$ position (restricted between 1.5 and 2.5)")
plt.ylabel("$\log(E_s(x_3))$")
plt.title(f"Log Riesz $s$-Energy for $x_3$ Positions between 1.5 and 2.5 (s=2)")
plt.legend()
plt.grid(True)
plt.show()