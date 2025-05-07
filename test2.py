import matplotlib.pyplot as plt
import numpy as np

x=np.arange(-500,500,0.1)
y=(1000-x)/(1+(1000-x)**2)+(-1000-x)/(1+(-1000-x)**2)

plt.plot(x,y)
plt.show()