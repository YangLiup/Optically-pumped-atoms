import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc


#注：本文的所有单位均采取国际单位制#

global h,c
h=6.626e-34
c=3e8
#计算失谐相关的耦合因子
def sigma(Gamma, nu,nu0):
    f=1/3
    re=2.82e-15
    sigma=re*f*c*(Gamma/2)/((nu-nu0)**2+(Gamma/2)**2)
    return sigma

def f(x):
    return x*np.exp(x)

W=inversefunc(f)

def photon_number(power,nu):
    return power/(h*nu)
#-----------------------碱金属原子种类-------------------------#
species='K'

#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
pN2=60
pHe=760*3
#-----------------------计算细节-------------------------#
f=1/3
c=3e8  #m/s
re=2.82e-15 #m

mol=6.02e23
R=8.314/mol
if species=='Rb':
    T=273.5+160   #开尔文
if species=='K':
    T=273.5+200   #开尔文
T0=273.5+20

nN2=133.32*pN2/(T0*R)
nHe=pHe/(T0*R)*133.32
if species=='K':
    nu_D1=c/(770.108e-9)                               #Hz
    p=10**(7.4077-4453/T)
    n=133.32*p/(T*R)
    Gamma=19.84*pHe*(T/T0)*1e6+18.98*pN2*(T/T0)*1e6       # Hz


if species=='Rb':
    nu_D1=377e12                               #Hz
    p=10**(2.881+4.312-4040/T)
    n=133.32*p/(T*R)                           # /m3
    amg=2.69e25
    Gamma=(18e3*nHe+17.8e3*nN2 )/amg*1e6       # Hz

Rsd=5000
nu=nu_D1+100e9
power=0.7 #W
area=15e-3*20e-3
I0=power/area
l=2e-2
z=np.arange(0,l,0.0001)

Rop10=I0*sigma(Gamma,nu,nu_D1)/(h*nu_D1)
Ropz1=Rsd*W(Rop10/Rsd*np.exp(Rop10/Rsd-n*sigma(Gamma,nu,nu_D1)*z))

#两束泵浦光对打
# Rop20=I0*sigma(Gamma,nu,nu_D1)/(h*nu_D1)
# Ropz2=Rsd*W(Rop20/Rsd*np.exp(Rop20/Rsd-n*sigma(Gamma,nu,nu_D1)*(-z+l)))

#镜子反射
Rop20=Ropz1[-1]
Ropz2=Rsd*W(Rop20/Rsd*np.exp(Rop20/Rsd-n*sigma(Gamma,nu,nu_D1)*(-z+l)))

Ropz=Ropz1+Ropz2
Pz=Ropz/(Ropz+Rsd)

plt.plot(z*1000,Pz)
plt.xlabel('z (mm)')
plt.ylabel('$P_z$')
plt.show()


