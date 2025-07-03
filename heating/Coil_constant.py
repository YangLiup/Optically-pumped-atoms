import numpy as np
import matplotlib.pyplot as plt
z0=20e-3
def single_coil_constant(a,b,z):
    mu0=4*np.pi*1e-7
    I=1 #安培
    B=mu0*I/(4*np.pi)*(4*a*b/(a**2+4*z**2)+4*a*b/(b**2+4*z**2))/np.sqrt(a**2+b**2+4*z**2)
    return B


def multi_coil_constant(n,z0,line_spacing):
    """
    n为匝数，z0为线圈距离气室中心的距离
    """
    # sign=np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]) # not carefully designed
    sign=np.array([1,-1,-1,1,   -1,1,1,-1,  -1,1,1,-1,  1,-1,-1,1]) # carefully designed
    total_B=0
    for i in np.arange(0,n,1):
        total_B=sign[i]*single_coil_constant(40e-3-i*line_spacing,40e-3-i*line_spacing,z0)+total_B
    return total_B

def two_layers_coil_constant(n,layer_spacing):
    return multi_coil_constant(n,z0,line_spacing)-multi_coil_constant(n,z0+layer_spacing,line_spacing)

# a=np.arange(0,100e-3,1e-3)
# b=a
# plt.plot(a,single_coil_constant(a,a,20e-3))
# plt.show()
z0=20e-3
layer_spacing=0.2e-3
line_spacing=0.4e-3
delta_voltage=40e-6
R= 100
n=8
delta_B=delta_voltage/R*two_layers_coil_constant(n,layer_spacing)
print(delta_B)

# """
# Johnson热电压噪声, nV量级
# """
# T=473  # K
# kB=1.38e-23 #J/K
# R=100
# delta_v=np.sqrt(4*kB*T*R)
# print(delta_v)