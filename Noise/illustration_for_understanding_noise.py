import numpy as np
import matplotlib.pyplot as plt
import scienceplots

O=np.arange(-5,5,0.01)
p=1/(np.sqrt(2*np.pi))*np.exp(-(O-0)**2/2)
p1=1/(np.sqrt(2*np.pi))*np.exp(-(O+1)**2/2)
p2=1/(np.sqrt(2*np.pi))*np.exp(-(O-1.5)**2/2)
# p3=1/(np.sqrt(2*np.pi))*np.exp(-(O+0.7)**2/2)
# p4=1/(np.sqrt(2*np.pi))*np.exp(-(O+1.5)**2/2)
# p5=1/(np.sqrt(2*np.pi))*np.exp(-(O+.5)**2/2)
# p6=1/(np.sqrt(2*np.pi))*np.exp(-(O+.2)**2/2)
# p7=1/(np.sqrt(2*np.pi))*np.exp(-(O+0.7)**2/2)
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(O, p)
    ax1.plot(O, p1,linestyle='dashed')
    ax1.plot(O, p2,linestyle='dashed')
    # ax1.plot(O, p3,linestyle='dashed')
    # ax1.plot(O, p7)
    ax1.set_xticks([0])
# # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax1.set_xticklabels(['$\hat O$'], fontsize=10)

    plt.annotate('p$(\hat O)$ given by $\\rho$', xy=(0, 1/(np.sqrt(2*np.pi))*np.exp(-(0-0)**2/2)), xytext=(1,  0.45),
            arrowprops=dict(arrowstyle='->', color='black'),fontsize=8)
    plt.annotate('$\langle \hat O \\rangle_c$', xy=(-1, 1/(np.sqrt(2*np.pi))*np.exp(-(0-0)**2/2)), xytext=(-3,  0.4),
            arrowprops=dict(arrowstyle='->', color='black'),fontsize=8)
    plt.annotate('$\langle \hat O \\rangle_c$', xy=(1.5, 1/(np.sqrt(2*np.pi))*np.exp(-(0-0)**2/2)), xytext=(2.5,  0.4),
            arrowprops=dict(arrowstyle='->', color='black'),fontsize=8)

    # ax1.set_xlim([0.01,1])
    ax1.set_ylim([0,.5])
    # ax1.set_xlabel('$O$', fontsize=10)
    ax1.set_ylabel('p$(\hat O)$', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )

    plt.savefig('picture.png', dpi=1000)

plt.show()