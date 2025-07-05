import numpy as np
import matplotlib.pyplot as plt
z0=20e-3
def single_coil_constant(a,b,z):
    """
    一个长a，宽b的矩形线圈框在轴线坐标轴z0处在单位电流下产生的磁场大小
    """
    mu0=4*np.pi*1e-7
    I=1 #安培
    B=mu0*I/(4*np.pi)*(4*a*b/(a**2+4*z**2)+4*a*b/(b**2+4*z**2))/np.sqrt(a**2+b**2+4*z**2)
    return B


def multi_coil_constant(n,z0,line_spacing):
    """
    n为匝数，z0为线圈距离气室中心的距离
    """
    # sign=np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]) # not carefully designed
    sign=np.array([1,-1,-1,1,   -1,1,1,-1,  -1,1,1,-1,  1,-1,-1,1,  -1,1,1,-1,  1,-1,-1,1,  1,-1,-1,1,   -1,1,1,-1]) # carefully designed
    total_B=0
    for i in np.arange(0,n,1):
        total_B=sign[i]*single_coil_constant(40e-3-i*line_spacing,40e-3-i*line_spacing,z0)+total_B
    return total_B

def two_layers_coil_constant(n,layer_spacing):
    """
    两层加热膜，膜与膜之间的距离为layer_spacing
    """
    return multi_coil_constant(n,z0,line_spacing)-multi_coil_constant(n,z0+layer_spacing,line_spacing)

# a=np.arange(0,100e-3,1e-3)
# b=a
# plt.plot(a,single_coil_constant(a,a,20e-3))
# plt.show()

# """
# 两层加热膜产生的磁噪声大小
# """
mu0=4*np.pi*1e-7
z0=20e-3
layer_spacing=0.2e-3
line_spacing=0.4e-3
delta_voltage=40e-6
R= 100
delta_current=delta_voltage/R
n=32
delta_B=delta_current*two_layers_coil_constant(n,layer_spacing)
print('两层',n,'匝方形线圈加热膜的磁噪声为', delta_B)

# """
# Johnson热电压噪声, nV量级
# """
T=473  # K
kB=1.38e-23 #J/K
R=100
delta_v=np.sqrt(4*kB*T*R)
print('Johnson电压为',delta_v)

# """
# 落单线的磁场大小
# """
l0=1e-3
delta_Bs=delta_current*mu0/(4*np.pi*z0)*(2*l0/np.sqrt(l0**2+z0**2))-delta_current*mu0/(4*np.pi*(z0+layer_spacing))*(2*l0/np.sqrt(l0**2+(z0+layer_spacing)**2))

print('落单线长度为',2*l0,'，磁噪声为',delta_Bs)