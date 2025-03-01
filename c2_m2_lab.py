# COULD NOT GET WORKING :(

import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import sympy as sp
from sympy import *

#======================================================================
# UNPACK THE DATA
#======================================================================

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]



#======================================================================
# INITIALIZE PARAMETERS
#======================================================================

v_var = 0.01  # translation velocity variance  
om_var = 0.01  # rotational velocity variance 
r_var = 0.1  # range measurements variance
b_var = 0.1  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x



#======================================================================
# CALCULATE JACOBIANS
#======================================================================

# H
#xl, xk, d, thk, yl, yk = sp.symbols('xl xk d thk yl yk')
#h = Matrix([[((xl-xk-d*cos(thk))**2 + (yl-yk-d*sin(thk))**2)**(0.5)], [atan2(yl-yk-d*sin(thk),xl-xk-d*cos(thk))-thk]])
#state = sp.Matrix([xk, yk, thk])
#H = h.jacobian(state)
#print(H)

# F
#xk, yk, thk, T = sp.symbols('xk yk thk T')
#x = Matrix([[xk],[yk],[thk]]) + T*Matrix([[cos(thk), 0], [sin(thk), 0], [0,1]]) @ (Matrix([[v_var],[om_var]]))
#state = sp.Matrix([xk,yk,thk])
#F = x.jacobian(state)
#print(F)



#======================================================================
# CORRECTION STEP
#======================================================================

def measurement_update(lk, rk, bk, P_check, x_check):

    # 1. Compute measurement Jacobian
    A = lk[0] - x_check[0] - d[0] * np.cos(x_check[2])
    B = lk[1] - x_check[1] - d[0] * np.sin(x_check[2])
    C = lk[0] - x_check[0]
    D = lk[1] - x_check[1]
    dist_2 = A**2 + B**2
    dist = np.sqrt(dist_2)
    
    H_k = np.array([[-A/dist, -B/dist, d*(C*np.sin(x_check[2]) - D*np.cos(x_check[2]))/dist], [B/dist_2, -A/dist_2, (C*d*np.cos(x_check[2])+D*d*np.sin(x_check[2])-C**2-D**2)/dist_2]])
    
    M_k = np.eye(2)
    # 2. Compute Kalman Gain
    K_k = P_check*H_k.T*(H_k*P_check*H_k.T + M_k*cov_y*M_k.T).inv
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_k_l = np.mat([[rk], [wraptopi(bk)]])
    y_k_l_predict = np.zeros([2, 1])
    y_k_l_predict[0] = dist
    y_k_l_predict[1] = np.arctan2(B, A) - x_check[2]
    y_k_l_predict[1] = wraptopi(y_k_l_predict[1])
    add = K_k*(y_k_l - y_k_l_predict)
    add = np.array(add).flatten()
    x_check = x_check + add
    x_check[2] = wraptopi(x_check[2])
    # 4. Correct covariance
    P_check = (np.eye(3) - K_k*H_k)*P_check
    
    return x_check, P_check
#======================================================================
# PREDICTION STEP
#======================================================================

#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    x_check = np.zeros((3,))
    x_check[0] = x_est[k-1, 0]
    x_check[1] = x_est[k-1, 1]
    x_check[2] = x_est[k-1, 2]
    
    P_check = P_est[k-1, :, :]
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    T_mat = delta_t * np.mat([[np.cos(x_check[2]), 0],
                              [np.sin(x_check[2]), 0],
                              [0, 1]])
    x_k = np.mat([[v[k]], [om[k]]])
    add = T_mat*x_k
    add = np.array(add).reshape((3,))
    x_check = x_check + add
    x_check[2] = wraptopi(x_check[2])
    # 2. Motion model jacobian with respect to last state
    F_km = np.mat([[1,0,-delta_t*v[k-1]*np.sin(x_check[2])],
                   [0,1,delta_t*v[k-1]*np.cos(x_check[2])],
                   [0,0,1]])
    # 3. Motion model jacobian with respect to noise
    L_km = np.mat([[delta_t*np.cos(x_check[2]), 0],
                   [delta_t*np.sin(x_check[2]), 0],
                   [0, delta_t]])
    # 4. Propagate uncertainty
    P_check = F_km*P_check*F_km.T + L_km*Q_km*L_km.T
    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check
    P_est[k, :, :] = P_check



#======================================================================
# PLOT
#======================================================================
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()
