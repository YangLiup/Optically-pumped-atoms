import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate

#计算失谐相关的耦合因子
def alpha0(Gamma, omega,omega0):
    f=1/3
    c=3e8
    re=2.82e-15
    alpha0=-re*f*c**2/(2*omega)/(omega-omega0+1j*Gamma)
    return alpha0

def photon_number(power):
    h=6.626e-34
    c=3e8
    lam=795e-9
    nu=c/lam
    # nu=c/(770.108e-9)
    return power/(h*nu)
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
kB=1.38e-23
R=8.314/mol
if species=='Rb':
    T=273.5+160
if species=='K':
    T=273.5+200
mHe=0.004/mol
mN2=0.028/mol
mK=0.039/mol
mRb=0.085/mol
T0=273.5+20
vHeK=np.sqrt(8*kB*T/np.pi/(mHe*mK/(mHe+mK)))
vHeRb=np.sqrt(8*kB*T/np.pi/(mHe*mRb/(mHe+mRb)))

vN2K=np.sqrt(8*kB*T/np.pi/(mN2*mK/(mN2+mK)))
vN2Rb=np.sqrt(8*kB*T/np.pi/(mN2*mRb/(mN2+mRb)))

vK=np.sqrt(16*kB*T/np.pi/mK)
vRb=np.sqrt(16*kB*T/np.pi/mRb)

nN2=133.32*pN2/(T0*R)
nHe=pHe/(T0*R)*133.32
if species=='K':
    nu_D1=c/(770.108e-9)
    omega0=2*np.pi*nu_D1
    k=2*np.pi*nu_D1/c
    p=10**(7.4077-4453/T)
    n=133.32*p/(T*R)
    Gamma=19.84*pHe*(T/T0)+18.98*pN2*(T/T0)       # MHz


if species=='Rb':
    nu_D1=377e12
    omega0=2*np.pi*nu_D1
    k=2*np.pi*nu_D1/c
    p=10**(2.881+4.312-4040/T)
    n=133.32*p/(T*R)
    amg=2.69e25
    Gamma=(18e3*nHe+17.8e3*nN2 )/amg      # MHz


omega=omega0
Je=1/2
l=np.arange(0,10,0.01)
Pz=0
phi=2*np.pi*n*k*alpha0(Gamma,omega,omega0)*l
theta=-np.pi*n*k*alpha0(Gamma,omega,omega0)*(11-4*Je*(Je+1))*Pz*l
I0=1e6
I=I0*np.exp(-2*phi.imag-2*theta.imag)

plt.plot(l,I)
plt.show()


