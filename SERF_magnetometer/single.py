import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def fun(X):
    global c,nu_D1,Gamma_D,Gamma_SD,delta_Bsp, Delta_nu,delta_Bph ,re

#-----------------------碱金属原子种类-------------------------#
    species='K'
#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
    pN2=60
    pHe=760*3
#-----------------------计算细节-------------------------#
    Gamma_pr=X[0]
    Rop=X[1]
    f=1/3
    c=3e8
    re=2.82e-15
    m=7
    gamma_e=2*np.pi*2.8*10**10  #Hz/T
    a=0.5e-2                      #m
    b=0.5e-2                       #m
    l=1.8e-2*m                     #m
    V=a*b*l

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
    vHe=np.sqrt(8*kB*T/np.pi/mHe)
    vN2=np.sqrt(8*kB*T/np.pi/mN2)
    vK=np.sqrt(8*kB*T/np.pi/mK)
    vRb=np.sqrt(8*kB*T/np.pi/mRb)

    nN2=133.32*pN2/(T0*R)
    nHe=pHe/(T0*R)*133.32
    if species=='K':
        nu_D1=c/(770.108e-9)
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
        nu_D1=377e12
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
    
    r=23e-1/2
    q=5
    Gamma_D=q*D*(np.pi/r)**2
    eta=0.5
    delta_Bsp=1/(gamma_e*np.sqrt(n*V)*Rop)*np.sqrt(4*(Rop+Gamma_pr+Gamma_SD+Gamma_D)**3)

    delta_Bph=1/(gamma_e*np.sqrt(n*V)*Rop)*np.sqrt(2*(Rop+Gamma_pr+Gamma_SD+Gamma_D)**4/(eta*Gamma_pr*OD))

    delta_B=np.sqrt(delta_Bsp**2+delta_Bph**2) # 目标函数
    return delta_B


delta_B=dual_annealing(fun,bounds=[[0,2000],[0,2000]])
print(delta_B)
fun(delta_B.x)

h=6.626e-34
Power_pump=delta_B.x[1]/re/c*3*(Delta_nu/2*1e6)*h*nu_D1
detuning=10e9
Power_probe=delta_B.x[0]/re/c*3*((Delta_nu/2*1e6)**2+detuning**2)/(Delta_nu/2*1e6)*h*nu_D1

print(delta_Bsp)
print(delta_Bph)

print(Power_pump)
print(Power_probe)



