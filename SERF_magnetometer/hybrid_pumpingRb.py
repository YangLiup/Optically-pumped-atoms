import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def fun(X):
    global c,nuK_D1,nuRb_D1,Gamma_DK,Gamma_DRb,Gamma_SDK,Gamma_DRb,delta_Bsp, Delta_nuK,Delta_nuRb,delta_Bph ,re,ODK,ODRb

#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
    pN2=30
    pHe=760*3
    m=1
#-----------------------待优化参数-------------------------#
    Gamma_pr=X[0]
    Rop=X[1]


#-----------------------常参数-------------------------#
    f=1/3
    c=3e8
    re=2.82e-15
    gamma_e=2*np.pi*2.8*10**10  #Hz/T
    mol=6.02e23
    kB=1.38e-23
    R=8.314/mol
    mHe=0.004/mol
    mN2=0.028/mol
    mK=0.039/mol
    mRb=0.085/mol
    amg=2.69e25

#-----------------------探测体积-------------------------#
    a=0.5e-2                      #m
    b=0.5e-2                       #m
    l=1.8e-2*m                     #m
    V=a*b*l

#----------------------温度-------------------------#
    T=273.5+160
    T0=273.5+20

#---------------------相对速度------------------------#
    vHeK=np.sqrt(8*kB*T/np.pi/(mHe*mK/(mHe+mK)))
    vHeRb=np.sqrt(8*kB*T/np.pi/(mHe*mRb/(mHe+mRb)))

    vN2K=np.sqrt(8*kB*T/np.pi/(mN2*mK/(mN2+mK)))
    vN2Rb=np.sqrt(8*kB*T/np.pi/(mN2*mRb/(mN2+mRb)))

    vK=np.sqrt(16*kB*T/np.pi/mK)
    vRb=np.sqrt(16*kB*T/np.pi/mRb)
    vKRb=np.sqrt(8*kB*T/np.pi/(mK*mRb/(mK+mRb)))

#---------------------缓冲气体数密度------------------------#
    nN2=133.32*pN2/(T0*R)
    nHe=pHe/(T0*R)*133.32

#---------------------D1线频率------------------------#
    nuK_D1=c/(770.108e-9)
    nuRb_D1=377e12

#---------------------碱金属蒸汽数密度------------------------#
    pRb=10**(2.881+4.312-4040/T)
    nRb=133.32*pRb/(T*R)
    nK=nRb/15

#---------------------光学线宽和OD------------------------#
    Delta_nuK=19.84*pHe*(T/T0)+18.98*pN2*(T/T0)       # MHz
    Delta_nuRb=(18e3*nHe+17.8e3*nN2 )/amg      # MHz

    ODK=2*re*c*f*nK*l/(Delta_nuK*1e6)
    ODRb=2*re*c*f*nRb*l/(Delta_nuRb*1e6)


#---------------------自旋破坏弛豫------------------------#
    Gamma_SDK=1e-19*nK+5e-29*nHe*vHeK+7.9e-27*nN2*vN2K+4.5e-22*nRb*vKRb
    Gamma_SDRb=9e-22*nRb*vRb+9e-28*nHe*vHeRb+1e-26*nN2*vN2Rb+4.5e-22*nK*vKRb

# -------------------------扩散弛豫-----------------------#
    D0_HeK=0.35
    D_HeK=D0_HeK*(760/pHe)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
    D0_N2K=0.2
    D_N2K=D0_N2K*(760/pN2)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
    DK=1/(1/D_HeK+1/D_N2K)
        
    D0_HeRb=0.42
    D_HeRb=D0_HeRb*(760/pHe)*(T0/(273.5+27))**(3/2)*(T/T0)**(1/2)
    D0_N2Rb=0.159
    D_N2Rb=D0_N2Rb*(760/pN2)*(T0/(273.5+60))**(3/2)*(T/T0)**(1/2) 
    DRb=1/(1/D_HeRb+1/D_N2Rb)
    
    r=23e-1/2
    q=5
    Gamma_DRb=q*DRb*(np.pi/r)**2
    Gamma_DK=q*DK*(np.pi/r)**2

# ------------------------探测效率-----------------------#
    eta=0.8

# ------------------------自旋交换率-----------------------#
    Rse_Rb=nRb*1e-18*vKRb
    Rse_K=nK*1e-18*vKRb

#--------------------稳态极化------------------------#
    PRb0=Rse_K*Rop/((Rse_K+Gamma_pr+Gamma_SDRb+Gamma_DRb)*(Rop+Rse_Rb+Gamma_SDK+Gamma_DK)-Rse_Rb*Rse_K)


#--------------------噪声------------------------#
    delta_Bsp=1/(gamma_e*np.sqrt(nRb*V)*PRb0)*np.sqrt(4*(Rse_K+Gamma_pr+Gamma_SDRb+Gamma_DRb))

    delta_Bph=1/(gamma_e*np.sqrt(nRb*V)*PRb0)*np.sqrt(2*(Rse_K+Gamma_pr+Gamma_SDRb+Gamma_DRb)**2/(eta*Gamma_pr*ODRb))

    delta_B=np.sqrt(delta_Bsp**2+delta_Bph**2) # 目标函数
    return delta_B


delta_B=dual_annealing(fun,bounds=[[0,2000],[0,2000]])
print(delta_B)
fun(delta_B.x)

h=6.626e-34
Power_pump=delta_B.x[1]/re/c*3*(Delta_nuRb/2*1e6)*h*nuRb_D1
detuning=10e9
Power_probe=delta_B.x[0]/re/c*3*((Delta_nuK/2*1e6)**2+detuning**2)/(Delta_nuK/2*1e6)*h*nuK_D1

print(delta_Bsp)
print(delta_Bph)

print(Power_pump)
print(Power_probe)


