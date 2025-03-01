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
        t = model.sample_time
        model.delta = model.delta + w * t
        delta = model.delta
        model.beta = np.arctan(lr*np.tan(delta)/L)
        beta = model.beta
        theta_d = v * np.cos(beta) * np.tan(delta) / L
        model.theta = model.theta + theta_d * t
        theta = model.theta
        xc_d = v * np.cos(theta + beta)
        yc_d = v * np.sin(theta + beta)
        model.xc = model.xc + xc_d * t
        model.yc = model.yc + yc_d * t
        pass 

model = Bicycle()
sample_time = 0.01
time_end = 30
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_data = np.zeros_like(t_data)

# ==================================
#  Learner solution begins here
# ==================================
v_data[:] = 16 * np.pi / 15
w_data[0:20] = model.w_max
w_data[350:391] = -model.w_max
w_data[1855:1896] = model.w_max
# ==================================
#  Learner solution ends here
# ==================================

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(v_data[i], w_data[i])

plt.axis('equal')
plt.plot(x_data, y_data)
plt.show()
