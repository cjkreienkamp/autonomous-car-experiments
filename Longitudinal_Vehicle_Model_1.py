import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Vehicle():
    def __init__(self):
 
        # ==================================
        #  Parameters
        # ==================================
    
        #Throttle to engine torque
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002
        
        # Gear ratio, effective radius, mass + inertia
        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81
        
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01
        
        # Tire force 
        self.c = 10000
        self.F_max = 10000
        
        # State variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0
        
        self.sample_time = 0.01
        
    def reset(self):
        # reset state variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

class Vehicle(Vehicle):
    def step(self, throttle, alpha):
        # ==================================
        #  Implement vehicle model here
        # ==================================
        model.x += model.v * sample_time + model.a * sample_time**2 / 2
        model.v += model.a * sample_time
        model.w_e += model.w_e_dot * sample_time
        F_load = model.c_a * model.v**2 + model.c_r1 * model.v + model.m * model.g * np.sin(alpha)
        T_e = throttle * (model.a_0 + model.a_1 * model.w_e + model.a_2 * model.w_e**2)
        model.w_e_dot = (T_e - model.GR * model.r_e * F_load) / model.J_e
        w_w = model.GR * model.w_e
        s = (w_w * model.r_e - model.v) / model.v
        if s < 1:
            F_x = model.c * s
        else:
            F_x = model.F_max
        model.a = (F_x - F_load) / model.m
        pass

sample_time = 0.01
time_end = 100
model = Vehicle()

t_data = np.arange(0,time_end,sample_time)
v_data = np.zeros_like(t_data)

# throttle percentage between 0 and 1
throttle = 0.2

# incline angle (in radians)
alpha = 0

for i in range(t_data.shape[0]):
    v_data[i] = model.v
    model.step(throttle, alpha)
    
plt.plot(t_data, v_data)
plt.show()
