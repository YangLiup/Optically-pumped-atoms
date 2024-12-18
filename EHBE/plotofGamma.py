import matplotlib.pyplot as plt
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
import pandas as pd
import scienceplots

# DD1,z,PP1=Gamma(3/2,0.01,0.005,100000)
# DD2,zz,PP2=Gamma(5/2,0.01,0.001,100000)
# DD3,zzz,PP3=Gamma(7/2,0.01,0.001,100000)
z = np.array(pd.read_csv('data/dt=0.001/z.csv'))
PP1 = np.array(pd.read_csv('data/dt=0.001/PP1.csv'))
DD1 = np.array(pd.read_csv('data/dt=0.001/DD1.csv'))
zz = np.array(pd.read_csv('data/dt=0.001/zz.csv'))
PP2 = np.array(pd.read_csv('data/dt=0.001/PP2.csv'))
DD2 = np.array(pd.read_csv('data/dt=0.001/DD2.csv'))
zzz = np.array(pd.read_csv('data/dt=0.001/zzz.csv'))
PP3 = np.array(pd.read_csv('data/dt=0.001/PP3.csv'))
DD3 = np.array(pd.read_csv('data/dt=0.001/DD3.csv'))
q1=2*(3+PP1**2)/(1+PP1**2)
eta1=(q1+4)/(q1-4)
fp1 = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)
fm1 = 2*q1/(q1-4)#*(q1-4)/(q1+4)

q2 = 2 * (19 + 26 * PP2 ** 2 + 3 * PP2 ** 4) / (3 + 10 * PP2 ** 2 + 3 * PP2 ** 4)
eta2=(q2+6)/(q2-6)
fp2 =  (q2-6)**2*(q2+6)/(2*36*q2**3)#*(q2+6)/(q2-6)
fm2 =2*q2/(q2-6)#*(q2-6)/(q2+6)

q3 = 2 * (11 + 35 * PP3 ** 2 + 17 * PP3 ** 4 + PP3 ** 6) / (1 + 7 * PP3 ** 2 + 7 * PP3 ** 4 + PP3 ** 6)
eta3=(q3+8)/(q3-8)
fp3 = (q3-8)**2*(q3+8)/(2*64*q3**3)#*(q3+8)/(q3-8)
fm3 =  2*q3/(q3-8)#*(q3-8)/(q3+8)
epsilon1=(fp1/DD1-z/fm1)/(fp1/DD1+z/fm1)
epsilon2=(fp2/DD2-zz/fm2)/(fp2/DD2+zz/fm2)
epsilon3=(fp3/DD3-zzz/fm3)/(fp3/DD3+zzz/fm3)
kappa1=fp1/DD1
kappa2=fp2/DD2
kappa3=fp3/DD3

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(6.8, 4.2))
    ax4 = fig.add_subplot(224)
    pp,=ax4.plot(PP1,epsilon1)
    pp2,=ax4.plot(PP2,epsilon2)
    pp3,=ax4.plot(PP3,epsilon3)

    # ax4.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax4.set_xlim([0.,0.99])
    ax4.set_ylim([-0.0001,0.002])
    # ax4.set_ylim([-0.1,0.1])
    ax4.set_ylabel('$\\varepsilon$', fontsize=8)
    ax4.tick_params(axis='x', labelsize='0.9' )
    ax4.tick_params(axis='y', labelsize='0.9' )
    ax4.set_yticks([0,0.001,0.002]) # 设置刻度
    ax4.set_xlabel('$P$', fontsize=9)

    ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax4.text(0.45, 0.00202, '(d)',fontsize=8)

    ax3 = fig.add_subplot(223)
    ax3.plot(PP1,fp1/DD1)
    ax3.plot(PP2,fp2/DD2)
    ax3.plot(PP3,fp3/DD3)
    ax3.set_xlabel('$P$', fontsize=9)



    # ax4.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax3.set_xlim([0.,0.99])
    # ax4.set_ylim([-0.1,0.1])
    ax3.set_ylabel('$\\tilde{\kappa}^+$', fontsize=8)
    ax3.tick_params(axis='x', labelsize='0.9' )
    ax3.tick_params(axis='y', labelsize='0.9' )
    ax3.text(0.45, 0.234, '(c)',fontsize=8)
    ax3.set_xlim([0.,1])
    # ax3.set_xlabel('$P$', fontsize=9)

    # ax3.axes.xaxis.set_ticklabels([])

    ax1 = fig.add_subplot(221)
    ax1.plot(PP1, DD1)
    ax1.plot(PP2, DD2)
    ax1.plot(PP3, DD3)

    # ax3.plot(PP1, xiao,linestyle='dotted')
    # ax3.plot(PP2, Gammap2,linestyle='dotted')
    # ax3.plot(PP3, Gammap3,linestyle='dotted')
    ax1.text(0.45, 0.049, '(a)',fontsize=8)
    ax1.set_xlim([0.003,1])
    # ax3.set_ylim([0,0.1])

    ax1.set_ylabel('$\\tilde{\Gamma}^+_{\perp}$ $(\omega_e^2/R_{\\rm{se}})$', fontsize=8)
    ax1.tick_params(axis='x', labelsize='0.9' )
    ax1.tick_params(axis='y', labelsize='0.9' )
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(222)
    p21=ax2.plot(PP1, z)
    p22=ax2.plot(PP2, zz)
    p23=ax2.plot(PP3, zzz)


    # ax3.plot(PP1, Gammam1,linestyle='dotted')
    # ax3.plot(PP2, Gammam2,linestyle='dotted')
    # ax3.plot(PP3, Gammam3,linestyle='dotted')
    ax4.legend([pp, pp2, pp3],["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='upper right', prop={'size': 8})
    ax2.set_ylabel('$\\tilde{\Gamma}_{\perp}^- $ ($R_{\\rm{se}}$) ', fontsize=8)
    ax2.tick_params(axis='x', labelsize='0.9' )
    ax2.tick_params(axis='y', labelsize='0.9' )
    ax2.text(0.45,0.925, '(b)',fontsize=8)
    # p24=ax3.plot(PP, h*np.ones(bound),linestyle='dotted')
    # p25=ax3.plot(PP, hh*np.ones(bound),linestyle='dotted')
    # p26=ax3.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    ax2.set_xlim([0.,0.99])
    ax4.set_xlim([0,1])
    ax2.axes.xaxis.set_ticklabels([])
    ax1.tick_params(axis='both', which='major', labelsize=9, colors='black')
    ax1.tick_params(axis='both', which='minor', labelsize=9, colors='black')
    ax2.tick_params(axis='both', which='major', labelsize=9, colors='black')
    ax2.tick_params(axis='both', which='minor', labelsize=9, colors='black')
    ax3.tick_params(axis='both', which='major', labelsize=9, colors='black')
    ax3.tick_params(axis='both', which='minor', labelsize=9, colors='black')
    ax4.tick_params(axis='both', which='major', labelsize=9, colors='black')
    ax4.tick_params(axis='both', which='minor', labelsize=9, colors='black')
    # ax1.tick_params(labelsize='1')
    # ax2.tick_params(labelsize='1')
    # ax3.tick_params(labelsize='1')
    # ax4.tick_params(labelsize='1')



    # ax3.set_ylim([0.5,1])

plt.savefig('Fig1_Gamma.png', dpi=1000)
plt.show()