import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate
from pynverse import inversefunc


global h,c
h=6.626e-34
c=3e8
#计算失谐相关的耦合因子
def sigma(Gamma, nu,nu_D1):
    f=1/3
    re=2.82e-15
    sigma=re*f*c*(Gamma/2)/((nu-nu_D1)**2+(Gamma/2)**2)
    return sigma

def f(x):
    return x*np.exp(x)

W=inversefunc(f)

def photon_number(power):
    nu_D1=c/(770.108e-9)
    # nu=c/(770.108e-9)
    return power/(h*nu_D1)
#-----------------------碱金属原子种类-------------------------#
species='K'

#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
pN2=60
pHe=760*3
#-----------------------计算细节-------------------------#
f=1/3
c=3e8
re=2.82e-15

mol=6.02e23
R=8.314/mol
if species=='Rb':
    T=273.5+160
if species=='K':
    T=273.5+200
T0=273.5+20

nN2=133.32*pN2/(T0*R)
nHe=pHe/(T0*R)*133.32
if species=='K':
    nu_D1=c/(770.108e-9)
    p=10**(7.4077-4453/T)
    n=133.32*p/(T*R)
    Gamma=19.84*pHe*(T/T0)*1e6+18.98*pN2*(T/T0)*1e6       # Hz


if species=='Rb':
    nu_D1=377e12
    p=10**(2.881+4.312-4040/T)
    n=133.32*p/(T*R)
    amg=2.69e25
    Gamma=(18e3*nHe+17.8e3*nN2 )/amg*1e6       # Hz

Rrel=1000
nu=nu_D1+100e9
power=0.5 #W
area=15e-3*20e-3
I0=power/area
l=2e-2
z=np.arange(0,l,0.0001)

Rop10=I0*sigma(Gamma,nu,nu_D1)/(h*nu_D1)
Ropz1=Rrel*W(Rop10/Rrel*np.exp(Rop10/Rrel-n*sigma(Gamma,nu,nu_D1)*z))

#两束泵浦光对打
# Rop20=I0*sigma(Gamma,nu,nu_D1)/(h*nu_D1)
# Ropz2=Rrel*W(Rop20/Rrel*np.exp(Rop20/Rrel-n*sigma(Gamma,nu,nu_D1)*(-z+l)))

#镜子反射
Rop20=Ropz1[-1]
Ropz2=Rrel*W(Rop20/Rrel*np.exp(Rop20/Rrel-n*sigma(Gamma,nu,nu_D1)*(-z+l)))

Ropz=Ropz1+Ropz2
Pz=Ropz/(Ropz+Rrel)
# omega=omega0+100e9
# Je=1/2
# 
# Sz=0.4
# phi=2*np.pi*n*k*alpha0(Gamma,omega,omega0)*l
# theta=-2*np.pi*n*k*alpha0(Gamma,omega,omega0)*(11-4*Je*(Je+1))/4*Sz*l
# # rate=-4*np.pi*n*k*(1-(11-4*Je*(Je+1))/4*Sz)*alpha0(Gamma,omega,omega0).imag*l
# I=I0*np.exp(-2*phi.imag-2*theta.imag)
plt.plot(z,Pz)
plt.show()


