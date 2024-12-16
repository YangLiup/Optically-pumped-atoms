import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def fun(X):
#-----------------------碱金属原子种类-------------------------#
    species='Rb'
#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#

#-----------------------计算细节-------------------------#
    Gamma_pr=X[0]
    Rop=X[1]
    f=1/3
    c=3e8
    re=2.82e-15
    m=1
    gamma_e=2*np.pi*2.8*10**10  #Hz/T
    a=0.5e-2                       #m
    b=0.5e-2                       #m
    l=2.5e-2*m                     #m
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
        Delta_nu=19.84*pHe+18.98*pN2       # MHz
        OD=2*re*c*f*n*l/(Delta_nu*1e6)
        Gamma_SD=1e-22*n*vK+8e-29*nHe*vHe+7.9e-27*nN2*vN2
        # -------------------考虑扩散弛豫--------------#
        D0_He=0.35
        D_He=D0_He*(760/pHe)*(T0/273.5)**(3/2)*(T/T0)**(1/2)
        D0_N2=0.2
        D_N2=D0_N2*(760/pN2)*(T0/273.5)**(3/2)*(T/T0)**(1/2)
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
    
    r=18e-1/2
    q=5
    Gamma_D=q*D*(np.pi/r)**2

    delta_B=1/(gamma_e*np.sqrt(n*V)*Rop)*np.sqrt(4*(Rop+Gamma_pr+Gamma_SD+Gamma_D)**3+2*(Rop+Gamma_pr+Gamma_SD+Gamma_D)**4/(Gamma_pr*OD))# 目标函数

    return delta_B

pHe_range=np.arange(100,round(760*3),10)
pN2_range=np.arange(1,round(901),200)

z=np.zeros((len(pN2_range),len(pHe_range)))
i=0
for pN2 in tqdm(pN2_range):
    j=0
    for pHe in pHe_range:
        delta_B=dual_annealing(fun,bounds=[[0,1000],[0,1000]])
        z[i][j]=delta_B.fun
        j=j+1
    i=i+1

fig = plt.figure()
ax1 = fig.add_subplot()
for i in np.arange(0,5,1):
    ax1.plot(pHe_range, z[i,:])
    
ax1.legend(["PN2$=1$ Torr", "PN2$=201$ Torr", "PN2$=401$ Torr", "PN2$=601$ Torr", "PN2$=801$ Torr"],
               loc='lower right', prop={'size': 8})
ax1.set_xlabel('PHe (Torr)', fontsize=10)
ax1.set_ylabel('$\delta B\; (T/\sqrt{Hz})$', fontsize=10)

plt.savefig('sensitivity.png', dpi=1000)




