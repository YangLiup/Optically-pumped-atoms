import numpy as np

"""
长度单位均为mm
"""

global mu,L,R,R_avg,t,D,Ga,Gr

D=240
L=586.4
Ga,Gr=10,10
t1,t2,t3,t4,t5=1,1.3,1.3,1.5,2
t=np.array([0,t1,t2,t3,t4,t5])

mu_r1,mu_r2,mu_r3,mu_r4,mu_r5=7000,7000,7500,10000,7500
mu_r=np.array([0,mu_r1,mu_r2,mu_r3,mu_r4,mu_r5])

R_avg=np.zeros(6)
L_avg=np.zeros(6)


for j in np.arange(1,6,1):
    R_avg[j]=D/2+(j-1)*Gr+t[j]/2
    for i in np.arange(1,j,1):
        R_avg[j]=R_avg[j]+t[i]

for j in np.arange(1,6,1):
    L_avg[j]=L+2*(j-1)*Ga+t[j]
    for i in np.arange(1,j,1):
        L_avg[j]=L_avg[j]+2*t[i]

L=L_avg+t
R=R_avg+t/2

def transverse_shielding_factor(i):
    """S_T[i] = mu[i] * t[i] / (2 * R_avg[i])"""
    return mu_r[i] * t[i] / (2 * R_avg[i])

def total_transverse_shielding_factor(n):
    """S_T_tot"""
    S_T_tot = 1.0
    for i in np.arange(1,n,1):
        ratio = R_avg[i] / R_avg[i+1]
        S_T_tot = S_T_tot*transverse_shielding_factor(i)* transverse_shielding_factor(n)* (1 - ratio**2)
    return S_T_tot

def N(i):
    """N_i"""
    ratio = 0.5 * L[i] / R[i]
    return -0.048 / np.sqrt(ratio) + 0.329 / ratio - 0.053 / ratio**2

def f(i):
    """f_i = 1 + L_i / (200 * R_i)"""
    return 1 + L[i] / (200 * R[i])

def axial_shielding_factor(i):
    """S_T[i] = mu[i] * t[i] / (2 * R_avg[i])"""
    return (1+4*N(i)*transverse_shielding_factor(i))/f(i)

def total_axial_shielding_factor(n):
    """S_T_tot"""
    S_A_tot = 1.0
    for i in np.arange(1,n,1):
        S_A_tot = S_A_tot*axial_shielding_factor(i)*axial_shielding_factor(n)* 5/(L[i+1]/2/R[i+1])*(2*R[i]+2*R[i+1]-t[i+1])/(4*L[i]+(R[i]+R[i+1]-t[i+1]))*(1-(2*R[i]/(2*R[i+1]-t[i+1]))**2)*f(i)/(4*N(i))
    return S_A_tot

print(total_axial_shielding_factor(5))
print(total_transverse_shielding_factor(5))