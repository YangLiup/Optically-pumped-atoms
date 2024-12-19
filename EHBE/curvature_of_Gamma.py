import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# z = np.array(pd.read_csv('data/dt=0.001/z.csv'))
# PP = np.array(pd.read_csv('data/dt=0.001/PP1.csv'))
# DD = np.array(pd.read_csv('data/dt=0.001/DD1.csv'))


# z = np.array(pd.read_csv('data/dt=0.001/zz.csv'))
# PP = np.array(pd.read_csv('data/dt=0.001/PP2.csv'))
# DD = np.array(pd.read_csv('data/dt=0.001/DD2.csv'))

z = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\zzz.csv'))
PP = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\PP3.csv'))
DD = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\DD3.csv'))


for i in np.arange(0,10,1):
    deleter=[n for n in range(0, len(PP), 2)]
    PP=np.delete(PP, deleter)
    z=np.delete(z, deleter)
    DD=np.delete(DD, deleter)

# eta=(5+3*PP**2)/(1-PP**2)
# q1=2*(3+PP**2)/(1+PP**2)
# fm = 2*q1/(q1-4)
# fp = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)

# q2 = 2 * (19 + 26 * PP ** 2 + 3 * PP ** 4) / (3 + 10 * PP ** 2 + 3 * PP ** 4)
# eta=(q2+6)/(q2-6)
# fp =  (q2-6)**2*(q2+6)/(2*36*q2**3)#*(q2+6)/(q2-6)
# fm=2*q2/(q2-6)#*(q2-6)/(q2+6)

q3 = 2 * (11 + 35 * PP ** 2 + 17 * PP ** 4 + PP ** 6) / (1 + 7 * PP ** 2 + 7 * PP ** 4 + PP ** 6)
eta=(q3+8)/(q3-8)
fp = (q3-8)**2*(q3+8)/(2*64*q3**3)#*(q3+8)/(q3-8)
fm =  2*q3/(q3-8)#*(q3-8)/(q3+8)

ddz=np.zeros(len(z)-2)
for i in np.arange(0,len(z)-2,1):
    ddz[i]=((z[i+1]-z[i])/(PP[i+1]-PP[i])-(z[i+2]-z[i+1])/(PP[i+2]-PP[i+1]))/(PP[i+2]-PP[i])

plt.figure()
plt.plot(PP[10000:len(z)-2],ddz[10000:len(z)-2])
plt.show()
