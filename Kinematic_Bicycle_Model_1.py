import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22
        
        self.sample_time = 0.01
        
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

class Bicycle(Bicycle):
    def step(self, v, w):
        L = model.L
        lr = model.lr
        delta = model.delta
        model.beta = np.arctan(lr*np.tan(delta)/L)
        beta = model.beta
        theta_d = v * np.cos(beta) * np.tan(delta) / L
        model.theta = model.theta + theta_d * model.sample_time
        theta = model.theta
        xc_d = v * np.cos(theta + beta)
        yc_d = v * np.sin(theta + beta)
        model.xc = model.xc + xc_d * model.sample_time
        model.yc = model.yc + yc_d * model.sample_time
        pass 


sample_time = 0.01
time_end = 20
model = Bicycle()
model.reset()
   
# set delta directly
model.delta = np.arctan(2/10)
        
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
        
for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    
    if model.delta < np.arctan(2/10):
        model.step(np.pi, model.w_max)
    else:
        model.step(np.pi, 0)
            
plt.axis('equal')
plt.plot(x_data, y_data,label='Learner Model')
plt.legend()
plt.show()
