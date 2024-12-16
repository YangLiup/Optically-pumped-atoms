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
def chia(delta_nv):
    Gammap=0.06e9       #压强展宽GHz
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
    return power/(h*nu)

def Gamma_pr(delta_nv,power):
    Gammap=0.06e9       #压强展宽GHz
    Gammad=0.5e9        #多普勒展宽GHz
    re=2.83e-15
    c=3e8
    fD1=1/3
    sigma=np.pi*re*c*fD1/4*(voigt_profile(delta_nv+2.3e9,Gammad,Gammap).real/4+3/4*voigt_profile(delta_nv+3.1e9,Gammad,Gammap).real)
    Phi=photon_number(power)
    return Phi*sigma

#优化
def fun(X):
    Power=X[0]
    Detuning=X[1]
    tau=X[2]
    def F_error(tau): 
        fArea,err = integrate.quad(lambda t: (1-t/tau)*np.exp(-Gamma2*t),0,tau)
        return fArea
    Phi=photon_number(Power)
    n=3e17   #/cm^3
    l=2.4e-2      #cm
    A=6e-6     #cm^2
    N_at=n*l*A
    gamma=2*np.pi*2.8*10**10
    chi=chia(Detuning)
    Gamma2=1000/30+Gamma_pr(Detuning,Power)
    sigmaF_tau=0.86/np.sqrt(N_at)
    Fx0=0.2
    sigma=1/(gamma*Fx0*tau*np.exp(-Gamma2*tau))*np.sqrt(4*F_error(tau)/N_at+1/(2*chi**2*n**2*l**2*Phi))
    return sigma

sigma=dual_annealing(fun,bounds=[[0,1e-3],[1e8,1e10],[0,1e-1]])
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
