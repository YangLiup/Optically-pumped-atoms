import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def relaxation_rate(pN2):
#-----------------------碱金属原子种类-------------------------#
    species='K'
#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
    pHe=0.00001
#-----------------------计算细节-------------------------#
    Gamma_pr=0
    Rop=0
    f=1/3
    c=3e8
    re=2.82e-15
    m=1
    gamma_e=2*np.pi*2.8*10**10  #Hz/T
    a=0.5e-2                       #m
    b=0.5e-2                       #m
    l=1.8e-2*m                     #m
    V=a*b*l/m

    mol=6.02e23
    kB=1.38e-23
    R=8.314/mol
    T=273.5+200
    mHe=0.004/mol
    mN2=0.028/mol
    mK=0.039/mol
    mRb=0.085/mol
    T0=273.5+20
    vHe=np.sqrt(8*kB*T/np.pi/mHe)
    vN2=np.sqrt(8*kB*T/np.pi/mN2)
    vK=np.sqrt(8*kB*T/np.pi/mK)
    vRb=np.sqrt(8*kB*T/np.pi/mRb)

    nN2=133.32*pN2/(T0*R)
    nHe=pHe/(T0*R)*133.32

    if species=='K':
        p=10**(7.4077-4453/T)
        n=133.32*p/(T*R)
        Delta_nu=19.84*pHe*(T/T0)+18.98*pN2*(T/T0)       # MHz
        OD=2*re*c*f*n*l/(Delta_nu*1e6)
        # Gamma_SD=1e-22*n*vK+8e-29*nHe*vHe+7.9e-27*nN2*vN2
        Gamma_SD=1e-19*n+5e-29*nHe*vHe+7.9e-27*nN2*vN2
        # -------------------考虑扩散弛豫--------------#
        D0_He=0.35
        D_He=D0_He*(760/pHe)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
        D0_N2=0.2
        D_N2=D0_N2*(760/pN2)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
        D=1/(1/D_He+1/D_N2)
        
    if species=='Rb':
        p=10**(2.881+4.312-4040/T)
        n=133.32*p/(T*R)
        amg=2.69e25
        Delta_nu=(18e3*nHe+17.8e3*nN2 )/amg      # MHz
        OD=2*re*c*f*n*l/(Delta_nu*1e6)
        Gamma_SD=9e-22*n*vRb+9e-28*nHe*vHe+1e-26*nN2*vN2
        D0_He=0.42
        D_He=D0_He*(760/pHe)*(T0/(273.5+27))**(3/2)*(T/T0)**(1/2)
        D0_N2=0.159
        D_N2=D0_N2*(760/pN2)*(T0/(273.5+60))**(3/2)*(T/T0)**(1/2) 
        D=1/(1/D_He+1/D_N2)
    
    r=4e-1/2
    q=5
    Gamma_D=q*D*(np.pi/r)**2

    rate=Gamma_D+Gamma_SD
    return rate
Rate=np.array([])
for pN2 in np.arange(250,3000,1):
    rate=relaxation_rate(pN2)
    Rate=np.append(Rate,rate)

plt.figure()
plt.plot(np.arange(250,3000,1),Rate)
plt.show()
    







