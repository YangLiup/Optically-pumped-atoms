import numpy as np
import matplotlib.pyplot as plt


def N(L,R):
    N=-0.048/np.sqrt(0.5*L/R)+0.329/(0.5*L/R)-+0.053/(0.5*L/R)**2
    return N

t1=1 #mm
L1=100 #mm
R1=np.arange(0.01,600,1)
f1=1+L1/(200*R1)
ST1=1000*t1/(2*R1)
SA1=(1+4*N(L1,R1)*ST1)/f1

plt.plot(R1,SA1)