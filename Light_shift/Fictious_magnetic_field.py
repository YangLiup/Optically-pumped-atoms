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

pHe=1e-10 #Torr
pN2=1e-10 #Torr
Gammap=0.0001         # 压强展宽GHz (19.84*pHe*(T/T0)+18.98*pN2*(T/T0))*1e-3 
Gammad=0.5            # 多普勒展宽GHz


def photon_number(power):
    h=6.626e-34 
    c=3e8 # m/s
    nu=c/(770.108e-9)
    return power/(h*nu)

gamma_e=2*np.pi*2.8*1e10  #Hz/T
re=2.83e-15 # m
c=3e8       # m/s
fD1=1/3
power=10    #W/m^2
delta_nv_range=np.arange(-200,200,0.01)

B=np.zeros(len(delta_nv_range))
i=0
for delta_nv in delta_nv_range:
    B[i]=-np.pi*re*c*fD1*photon_number(power)/gamma_e*voigt_profile(delta_nv,Gammad,Gammap).imag*1e-9
    i=i+1

plt.plot(delta_nv_range,B)
plt.xlabel('detuning (GHz)')
plt.ylabel('$B_{\\text{ls}}$ (T)')
# plt.title('K')
# plt.text(300,
# 	0.7e-9,
# 	"Power=1 mW/cm$^2$ \n $p_{\\text{He}}=3\\times 760$ Torr \n $p_{\\text{N}_2}=60$ Torr"
# )