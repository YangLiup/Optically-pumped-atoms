import numpy as np

T=473  # K
kB=1.38e-23 #J/K
I=1e-3 #安培
f=1 # Hz
A=np.pi*(2e-3)**2 #m2
P=  1.17606785704218E-20 #W
delta_B=np.sqrt(4*kB*T*2*P)/(2*np.pi*f*A*I) #T
print(delta_B)

# T=473  # K
# kB=1.38e-23 #J/K
# mu0=4*np.pi*1e-7
# sigma=1.6*1e6
# t=0.1e-3
# r=10e-3
# delta_B=1/np.sqrt(2*np.pi)*mu0*np.sqrt(kB*T*sigma*t)/r #T
# print(delta_B)