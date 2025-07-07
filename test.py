import numpy as np
import matplotlib.pyplot as plt
w=np.arange(-100,100,0.1)
y=-(-w**2+w+1)+100/(w**2+0.1**2)
plt.plot(w,y)
plt.show()