import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from matplotlib.font_manager import FontProperties  # 导入FontProperties


DeltaB=(np.array(pd.read_csv('B.csv')).flatten()-1)*150e3/7*1e6 

B = np.array(pd.read_csv('single.csv')).flatten()/7*1e6

baseline=(np.array(pd.read_csv('baseline.csv')).flatten()-1)*150e3/7*1e6


fs = 1/(40e-3)   # 采样频率
dt = 1/fs    # 采样周期

res=fs/len(DeltaB)


def time_picture(t,y):
    with plt.style.context(['science','nature']):
        plt.plot(t,y)
        plt.xlabel('Time (s)')
        plt.ylabel('$\Delta B$ (fT)')
        plt.show()

def PSD(DeltaB,fs,res):
    point=1
    f, Pden = signal.periodogram(DeltaB, fs)
    Pden_filterd=signal.savgol_filter(Pden,window_length=51,polyorder=3)
    P_1=np.sqrt(np.sum(Pden[round(1/res)-point:round(1/res)+point])/(2*point))
    P_10=np.sqrt(np.sum(Pden[round(10/res)-point:round(10/res)+point])/(2*point))
    return f, Pden_filterd, P_1, P_10

# diff PSD计算
f_d, Pden_d, P_1, P_10=PSD(DeltaB,fs,res)
print(P_1/0.99,P_10/0.91)


# # baseline PSD计算
# f_b, Pden_b,P_1, P_10=PSD(baseline,fs,res)


# # sigle PSD计算
# f_s, Pden_s, P_1, P_10=PSD(B,fs,res)


plt.figure()
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
# p1,=plt.loglog(f_s,np.sqrt(Pden_s))
p2,=plt.plot(f_d,np.sqrt(Pden_d))
# p3,=plt.loglog(f_b,np.sqrt(Pden_b))
# plt.legend([p1,p2,p3], ["single", "diff","baseline"],
#             loc='lower left', prop={'size': 11})
# plt.xlim(0.2,11)
# plt.xlim([0.01,12])
# plt.ylim([0.1,1e5])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Noise spectrum ($\mathrm{fT/\sqrt{Hz}}$)')
plt.savefig('Sensitivity', dpi=600)
plt.show()










