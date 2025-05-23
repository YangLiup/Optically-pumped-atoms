import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from scipy import signal
from mpl_toolkits.mplot3d import axes3d
import time
"""

本文的频率单位均为kHz，时间单位均为ms

"""


T=20
dt=1e-5
t=np.arange(0,T,dt)

duration_op=2 #ms
amplitude_op = 50  #kHz#
frequency_op = 0.1 #kHz#
duty_op=duration_op*frequency_op
Rop = amplitude_op * signal.square(2 * np.pi * frequency_op * (t), duty=duty_op)+amplitude_op 

frequency_pi = 10 #kHz，integer是个整数，这是为了使得光泵浦和pi脉冲的相位稳定
amplitude_pi = 100  #kHz#
duration_pi=np.pi/(2*amplitude_pi)  #ms#
duty_pi=duration_pi*frequency_pi
theta_pi=np.pi/180*0.
phi_pi=0
omega_pix = (amplitude_pi * signal.square(2 * np.pi * frequency_pi * (t), duty=duty_pi)+amplitude_pi)*np.sin(theta_pi)*np.cos(phi_pi)
omega_piy = (amplitude_pi * signal.square(2 * np.pi * frequency_pi * (t), duty=duty_pi)+amplitude_pi)*np.sin(theta_pi)*np.sin(phi_pi)
omega_piz = (amplitude_pi * signal.square(2 * np.pi * frequency_pi * (t), duty=duty_pi)+amplitude_pi)*np.cos(theta_pi)

# for k in np.arange(0, round(T / dt), 1): 
#     if Rop[k]==2*amplitude_op:
#         omega_pix[k]=0
#         omega_piy[k]=0
#         omega_piz[k]=0


Gamma=0.01 
omega_0x=0.1e-1+omega_pix*1
omega_0y=0.1e-1+omega_piy*1
omega_0z=0.5e-1+omega_piz*1

n=round(T/dt)
Pxarray=np.zeros(n)
Pyarray=np.zeros(n)
Pzarray=np.zeros(n)

Px=0
Py=0
Pz=0
for i in trange(0,n,1):
    Pxarray[i]=Px
    Pyarray[i]=Py
    Pzarray[i]=Pz
    Px=(omega_0y[i]*Pz-omega_0z[i]*Py-Px*Gamma-Rop[i]*Px)*dt+Px
    Py=(-omega_0x[i]*Pz+omega_0z[i]*Px-Py*Gamma-Rop[i]*Py)*dt+Py
    Pz=(omega_0x[i]*Py-omega_0y[i]*Px-Pz*Gamma-Rop[i]*Pz+Rop[i])*dt+Pz




#绘制动态图像
fig = plt.figure(figsize=(3,5))
ax1 = fig.add_subplot(211)
ax1.plot(t,Pxarray)
ax1.set_ylabel('$P_y$')
ax1.set_xlabel('$t$', fontsize=8)


ax2 = fig.add_subplot(212)
for i in np.arange(0,n,200):
    # ax.plot(Pxarray[i],Pyarray[i],'bo')
    ax2.quiver(0,0,Pxarray[i],Pyarray[i],color=(1, 0, 0, 0.3),angles='xy', scale_units='xy', scale=1)
    ax2.set_xlim([-10e-4,10e-4])
    ax2.set_ylim([-10e-4,10e-4])
    ax2.set_xlabel('Px')
    ax2.set_xlabel('Py')
    ax2.grid()
    plt.pause(1e-4)
    ax2.cla()


#绘制静态图像
# fig = plt.figure()
# ax1 = fig.add_subplot(312)
# ax1.plot(t,Pxarray)
# ax1.set_ylabel('$P_y$', fontsize=8)
# ax1.tick_params(axis='both', which='major', labelsize=8)
# ax1.tick_params(axis='both', which='minor', labelsize=8)
# ax1.set_xticklabels([])

# ax2 = fig.add_subplot(312)
# ax2.plot([],[])
# ax2.plot([],[])
# ax2.plot(t,omega_piz)
# ax2.set_ylabel('$\omega_{\pi}$ (kHz)', fontsize=8)
# ax2.tick_params(axis='both', which='major', labelsize=8)
# ax2.tick_params(axis='both', which='minor', labelsize=8)
# ax2.set_xticklabels([])

# ax3 = fig.add_subplot(313)
# ax3.plot([],[])
# ax3.plot(t,Rop)
# ax3.set_ylabel('$R_{\\text{op}}$ (kHz)', fontsize=8)
# ax3.set_xlabel('t (ms)',fontsize=8)
# ax3.tick_params(axis='both', which='major', labelsize=8)
# ax3.tick_params(axis='both', which='minor', labelsize=8)

# plt.grid()
# plt.savefig('signal.png', dpi=1000)
# plt.show()