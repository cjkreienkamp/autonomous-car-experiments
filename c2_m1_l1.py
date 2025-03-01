# https://www.coursera.org/learn/state-estimation-localization-self-driving-cars/ungradedLab/oUqWi/lesson-1-practice-notebook-least-squares

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Store the voltage and current data as column vectors.
I = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
V = np.array([[1.23, 1.38, 2.06, 2.47, 3.17]]).T

# Define the H matix - what does it contain?
H = np.array([[1, 1, 1, 1, 1]]).T

# Now estimate the resistance parameter.
y = V/I
R = np.matmul(np.matmul(inv(np.matmul(H.T,H)), H.T), y)

print('The slope parameter of the best-fit line (i.e., the resistance) is:')
print(R[0,0])

# Plot
I_line = np.arange(0, 0.8, 0.1).reshape(8,1)
V_line = R*I_line

plt.scatter(I,V)
plt.plot(I_line, V_line)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()
