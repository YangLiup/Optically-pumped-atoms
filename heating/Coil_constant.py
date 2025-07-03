import numpy as np
import matplotlib.pyplot as plt
z0=20e-3
def single_coil_constant(a,b,z):
    mu0=4*np.pi*1e-7
    I=1 #安培
    B=mu0*I/(4*np.pi)*(4*a*b/(a**2+4*z**2)+4*a*b/(b**2+4*z**2))/np.sqrt(a**2+b**2+4*z**2)
    return B


def multi_coil_constant(n,z0):
    """
    n为匝数，z0为线圈距离气室中心的距离
    """
    # sign=np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]) # not carefully designed
    sign=np.array([1,-1,-1,1,   -1,1,1,-1,  -1,1,1,-1,  1,-1,-1,1]) # carefully designed
    total_B=0
    for i in np.arange(0,n,1):
        total_B=sign[i]*single_coil_constant(40e-3-i*(0.4e-3),40e-3-i*(0.4e-3),z0)+total_B
    return total_B

def two_layers_coil_constant(n):
    return multi_coil_constant(n,z0)-multi_coil_constant(n,z0+35e-6)

# a=np.arange(0,100e-3,1e-3)
# b=a
# plt.plot(a,single_coil_constant(a,a,20e-3))
# plt.show()

delta_voltage=40e-6
R= 100
n=2
delta_B=delta_voltage/R*two_layers_coil_constant(n)
print(delta_B)

# """
# Johnson热电压噪声, nV量级
# """
# T=473  # K
# kB=1.38e-23 #J/K
# R=100
# delta_v=np.sqrt(4*kB*T*R)
# print(delta_v)