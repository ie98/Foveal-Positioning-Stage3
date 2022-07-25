import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x,y = np.mgrid[0:100:101j,0:109:110j]  # 均值就是标注点 ， 如何根据边缘的位置计算方差
sigma = (25/2)
x1 = 55
y1 = 50
q = 1/(2 * np.pi * (sigma**2)) * np.exp(-((x-x1)**2+(y-y1)**2)/(2 * sigma**2))*100
min_q = np.min(q)
max_q = np.max(q)
q = (q-min_q)/(max_q-min_q)

p = -q+1
index = np.where(q == p)
print(q)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, q, rstride=1, cstride=1, cmap='rainbow',alpha = 0.9)

plt.savefig('./bagroundMask.png')
