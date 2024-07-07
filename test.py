import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from my_functions.master_relaxationrate import masterequation
from scipy.linalg import *
import numpy as np
from my_functions.test7_copy import gammam

# xiao's fast relaxation rate 
# h=3/4
# qq0=2*(19)/(3)
# hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

# qqq0=2*(11)/(1)
# hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

# Mr zhao's fast relaxation rate 

# varpsilon1=(5+P**2)/(1+P**2)
# q1 = 2 * (3 + P ** 2) / (1 + P ** 2)
# g1=P**2/(4*(1+P**2))
# h = (q1/8)/(varpsilon1)* 5

# varpsilon2=(35+42*P**2+3*P**4)/(3+10*P**2+3*P**4)
# q2 = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
# g2 = 8*P**2* (7 + 3 * P ** 2 ) / 27/(3 + 10 * P ** 2 + 3 * P ** 4)
# hh =(19/27-g2-g2) /varpsilon2*(35/3)

# varpsilon3=(21+63*P**2+27*P**4+P**6)/(1+7*P**2+7*P**4+P**6)
# q3 = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
# g3 = P**2* (21 + 30 * P ** 2 +5*P**4) / 16/(1+ 7 * P ** 2 + 7 * P ** 4+P**6)
# hhh =(22/32-2*g3)/varpsilon3*(21)


P1,Gamma1,PP1,DD1=masterequation(3/2)
# P2,Gamma2,PP2,DD2=masterequation(5/2)
# P3,Gamma3,PP3,DD3=masterequation(7/2)

q1=2*(3+PP1**2)/(1+PP1**2)
fp1 = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(111)
    ax1.plot(PP1,(fp1/DD1*(q1+4)/(q1-4)))
    # ax1.plot(PP2,(fp2/DD2))
    # ax1.plot(PP3,(fp3/DD3))
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot(PP1,1/8*6/(1+(q1+4)/(q1-4))*(q1+4)/(q1-4),linestyle='dashed')
    # ax1.plot(PP2,5/27*6/(1+(q1+4)/(q1-4))*(q1+4)/(q1-4),linestyle='dashed')
    # ax1.plot(PP3,(fp3/DD3),linestyle='dashed')
    # ax1.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax1.set_xlim([0.,0.999])
    ax1.set_ylim([0.,2])

    ax1.set_ylabel('Quality', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
plt.savefig('f_.png', dpi=1000)
# plt.style.use(['science','nature'])
# with plt.style.context(['science','nature']):
#   


#     # plt.ylim([0.65, 1])
#     # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     # plt.xlabel('Frequency (Hz)', fontsize=12)
#     # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)


#     # ax2.set_xlim([0,1000])
    

#     plt.savefig('Gamma_.png', dpi=1000)
# plt.show()