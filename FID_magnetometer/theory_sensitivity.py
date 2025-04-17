import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
from scipy.special import wofz
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate

#计算失谐相关的耦合因子
def voigt_profile(Delta, Gamma_G, Gamma_L):
    """
    Calculate the Voigt profile.

    Parameters:
        x (array-like): The x-values at which to calculate the profile.
        sigma (float): The Gaussian standard deviation.
        gamma (float): The Lorentzian full-width at half-maximum.

    Returns:
        array-like: The Voigt profile values at the specified x-values.
    """
    z = 2*np.sqrt(np.log(2))*(Delta+1j*Gamma_L/2)/Gamma_G
    v = wofz(z) * 2*np.sqrt(np.log(2)/np.pi)/Gamma_G
    return v




#delta_nv为失谐
def chia(delta_nv,pHe,pN2,T,T0):
    Gammap=(19.84*pHe*(T/T0)+18.98*pN2*(T/T0))*1e6       # Hz
    # Gammap=0.06e9       #压强展宽GHz
    Gammad=0.5e9        #多普勒展宽GHz
    re=2.83e-15
    c=3e8
    fD1=1/3
    chia=np.pi*re*c*fD1/4*(voigt_profile(delta_nv+2.3e9,Gammad,Gammap).imag/4+3/4*voigt_profile(delta_nv+3.1e9,Gammad,Gammap).imag)
    return chia

def photon_number(power):
    h=6.626e-34
    c=3e8
    lam=795e-9
    nu=c/lam
    # nu=c/(770.108e-9)
    return power/(h*nu)

def Gamma_pr(delta_nv,power,pHe,pN2,T,T0):
    Gammap=(19.84*pHe*(T/T0)+18.98*pN2*(T/T0))*1e6    #压强展宽Hz
    Gammad=0.5e9        #多普勒展宽Hz
    re=2.83e-15
    c=3e8
    fD1=1/3
    sigma=np.pi*re*c*fD1/4*(voigt_profile(delta_nv+2.3e9,Gammad,Gammap).real/4+3/4*voigt_profile(delta_nv+3.1e9,Gammad,Gammap).real)
    Phi=photon_number(power)
    return Phi*sigma

#优化
def fun(X):
    global n, vRb,vK,Gamma_se
    Power=X[0]
    Detuning=X[1]
    tau=X[2]
    species='K'
#-----------------------缓冲气体和淬灭气体的气压(室温时）-------------------------#
    pN2=60
    pHe=760*3
#-----------------------计算细节-------------------------#
    f=1/3
    c=3e8
    re=2.82e-15
    m=1
    gamma_e=2*np.pi*2.8*10**10  #Hz/T
    a=0.5e-2                      #m
    b=0.5e-2                       #m
    l=1.8e-2                     #m
    A=a*b     #m^2
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
        p=10**(7.4077-4453/T)
        n=133.32*p/(T*R)
        Delta_nu=19.84*pHe*(T/T0)+18.98*pN2*(T/T0)       # MHz
        OD=2*re*c*f*n*l/(Delta_nu*1e6)
        # Gamma_SD=1e-22*n*vK+8e-29*nHe*vHe+7.9e-27*nN2*vN2
        Gamma_SD=1e-19*n+5e-29*nHe*vHeK+7.9e-27*nN2*vN2K

        # -------------------考虑扩散弛豫--------------#
        D0_He=0.35
        D_He=D0_He*(760/pHe)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
        D0_N2=0.2
        D_N2=D0_N2*(760/pN2)*pow(T0/273.5,3/2)*np.sqrt(T/T0)
        D=1/(1/D_He+1/D_N2)
        Gamma_se=n*1.5e-18*vK
    if species=='Rb':
        nu_D1=377e12
        p=10**(2.881+4.312-4040/T)
        n=133.32*p/(T*R)
        amg=2.69e25
        Delta_nu=(18e3*nHe+17.8e3*nN2 )/amg      # MHz
        OD=2*re*c*f*n*l/(Delta_nu*1e6)
        Gamma_SD=9e-22*n*vRb+9e-28*nHe*vHeRb+1e-26*nN2*vN2Rb
        D0_He=0.42
        D_He=D0_He*(760/pHe)*(T0/(273.5+27))**(3/2)*(T/T0)**(1/2)
        D0_N2=0.159
        D_N2=D0_N2*(760/pN2)*(T0/(273.5+60))**(3/2)*(T/T0)**(1/2) 
        D=1/(1/D_He+1/D_N2)
        Gamma_se=n*1.9e-18*vRb

    
    r=23e-1/2
    q=4
    Gamma_D=q*D*(np.pi/r)**2

    def F_error(tau): 
        fArea,err = integrate.quad(lambda t: (1-t/tau)*np.exp(-Gamma2*t),0,tau)
        return fArea
    Phi=photon_number(Power)
    N_at=n*l*A
    chi=chia(Detuning,pHe,pN2,T,T0)
    Gamma2=Gamma_SD+Gamma_se+Gamma_D+Gamma_pr(Detuning,Power,pHe,pN2,T,T0)
    sigmaF_tau=0.86/np.sqrt(N_at)
    Sx0=1/2
    sigma=1/(gamma_e*Sx0*tau*np.exp(-Gamma2*tau))*np.sqrt(F_error(tau)/N_at/4+1/(2*chi**2*n**2*l**2*16*Phi))
    return sigma

sigma=dual_annealing(fun,bounds=[[0.,5],[5e9,100e9],[0,0.05]], maxiter=10000)
print(sigma)

#计算
# delta_nu=1.6e9
# power=1e-3
# n=1.5e11   #/cm^3
# l=2.4      #cm
# A=6e-2     #cm^2
# Nat=n*A*l
# tau=0.01
# Sz0=0.05
# Gammarel=60
# gamma=2*np.pi*7e-6
# phi0=0.0135
# delta_ph=np.sqrt(2)/(gamma*np.sqrt(photon_number(power))*tau*phi0)
# # delta=1/(-gamma*chia(delta_nu)*Nat*np.sqrt(2*photon_number(power))*tau*np.exp(-Gammarel*tau)*Sz0)
# print(delta_ph)
# delta_sp=2/(gamma*np.sqrt(Nat*tau))
# print(delta_sp)
