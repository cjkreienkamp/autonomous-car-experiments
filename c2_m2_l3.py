import numpy as np

# Set parameters
dt = 0.5
u_k = -2
y_k = np.pi/6
S = 20
D = 40

# Initial estimates
x_k = np.array([[0],[5]])
P_k = np.array([[0.01,0],[0,1]])
w_k = 0
Q_k = np.identity(2)*0.1
v_k = 0
R_k = 0.01

# Initial Jacobians
F_k = np.array([[1,dt],[0,1]])
L_k = np.identity(2)
M_k = 1


## ITERATION

# State Estimation (Prediction)
x_k_down = np.array([[1,dt],[0,1]]).dot(x_k) + np.array([[0],[dt]]).dot(u_k) + w_k
P_k = F_k.dot(P_k).dot(F_k.T) + L_k.dot(Q_k).dot(L_k.T)

# Measurement Model (Correcction)
filler = S/((D-x_k_down)**2 + S**2)
H_k = np.array([[S/((D-x_k_down[0])**2 + S**2), 0]], dtype=float)
K_k = P_k.dot(H_k.T)*1/(H_k.dot(P_k).dot(H_k.T) + M_k*R_k*M_k)
x_k_up = x_k_down + K_k*(y_k - np.arctan(S/(D-x_k_down[0])) + v_k)
P_k = (np.identity(2) - K_k.dot(H_k)).dot(P_k)

# Print
print ('P_1',P_k)
