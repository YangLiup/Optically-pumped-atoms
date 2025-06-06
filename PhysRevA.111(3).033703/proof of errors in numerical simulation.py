import matplotlib.pyplot as plt
from qutip import *
from my_functions.Relaxation_rate_calculation_by_DME import Gamma
from scipy.linalg import *
import numpy as np

DD1,z,PP1=Gamma(3/2,0.05,0.01,5000)
DD2,zz,PP2=Gamma(3/2,0.01,0.01,50000)


fig = plt.figure()
plt.rc('font',family='Times New Roman')
ax1 = fig.add_subplot(111)
ax1.plot(PP1,(DD1)/np.max(DD1),linewidth=0.5)
ax1.plot(PP2,(DD2)/np.max(DD1),linewidth=0.5)



plt.savefig('Gamma_.png', dpi=1000)
plt.show()