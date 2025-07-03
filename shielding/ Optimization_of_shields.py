import numpy as np
import matplotlib.pyplot as plt
"""
长度单位均为mm
"""

global mu,L,R,R_avg,t,D,Ga,Gr

def ST(n,Ga,L,D):
    # D=300
    # L=450
    Gr=Ga
    t1,t2,t3,t4,t5,t6=1,1,1,1,1,1
    t=np.array([0,t1,t2,t3,t4,t5,t6])

    mu_r1,mu_r2,mu_r3,mu_r4,mu_r5,mu_r6=20000,20000,20000,20000,20000,20000
    mu_r=np.array([0,mu_r1,mu_r2,mu_r3,mu_r4,mu_r5,mu_r6])

    R_avg=np.zeros(7)
    L_avg=np.zeros(7)


    for j in np.arange(1,7,1):
        R_avg[j]=D/2+(j-1)*Gr+t[j]/2
        for i in np.arange(1,j,1):
            R_avg[j]=R_avg[j]+t[i]

    for j in np.arange(1,7,1):
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
            S_T_tot = S_T_tot*transverse_shielding_factor(i)* (1 - ratio**2)
        return S_T_tot*transverse_shielding_factor(n)
    
    y=total_transverse_shielding_factor(n)
    # def N(i):
    #     """N_i"""
    #     ratio = 0.5 * L[i] / R[i]
    #     return -0.048 / np.sqrt(ratio) + 0.329 / ratio - 0.053 / ratio**2

    # def f(i):
    #     """f_i = 1 + L_i / (200 * R_i)"""
    #     return 1 + L[i] / (200 * R[i])

    # def axial_shielding_factor(i):
    #     """S_T[i] = mu[i] * t[i] / (2 * R_avg[i])"""
    #     return (1+4*N(i)*transverse_shielding_factor(i))/f(i)

    # # def total_axial_shielding_factor(n):
    # #     """S_T_tot"""
    # #     S_A_tot = 1.0
    # #     for i in np.arange(1,n,1):
    # #         ratio = L_avg[i] / L_avg[i+1]
    # #         S_A_tot = S_A_tot*axial_shielding_factor(i)* (1 - ratio**2)
    # #     return S_A_tot*axial_shielding_factor(n)

    # def total_axial_shielding_factor(n):
    #     """S_T_tot"""
    #     S_A_tot = 1.0
    #     for i in np.arange(1,n,1):
    #         S_A_tot = S_A_tot*axial_shielding_factor(i)* 5/(L[i+1]/2/R[i+1])*(2*R[i]+2*R[i+1]-t[i+1])/(4*L[i]+(R[i]+R[i+1]-t[i+1]))*(1-(2*R[i]/(2*R[i+1]-t[i+1]))**2)*f(i)/(4*N(i+1))
    #     return S_A_tot*axial_shielding_factor(n)
    return y

Ga_list=np.array([10,15,20,30,35,40,45,50,60,65,70,75,80])
i=0
factor=np.zeros(len(Ga_list))
for Ga in Ga_list:
    factor[i]=ST(6,Ga,405,270)
    i=i+1
plt.scatter(Ga_list,factor)
plt.xlabel('Air gap (mm)')
plt.ylabel('$S^T$')
plt.title('$L_0=405$ mm, $D_0=270$ mm')
plt.savefig('ST_Ga')
plt.show()

# L_list=np.array([450,460,470,480,490,500,510,520,530,540,550])
# i=0
# factor=np.zeros(len(L_list))
# for L in L_list:
#     factor[i]=ST(5,15,L,300)
#     i=i+1
# plt.scatter(L_list,factor)
# plt.xlabel('L (mm)')
# plt.ylabel('$S^T$')
# plt.title('Ga=15 mm, D=300 mm')
# plt.savefig('ST_L')
# plt.show()


# D_list=np.array([300,320,340,360,380,400,420,440,460,480,500])
# i=0
# factor=np.zeros(len(D_list))
# for D in D_list:
#     factor[i]=ST(5,15,450,D)
#     i=i+1
# plt.scatter(D_list,factor)
# plt.xlabel('D (mm)')
# plt.ylabel('$S^T$')
# plt.title('Ga=15 mm, L=450 mm')
# plt.savefig('ST_D')
# plt.show()