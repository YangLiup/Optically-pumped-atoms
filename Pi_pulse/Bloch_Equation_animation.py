import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy import signal
from tqdm import trange
from ipywidgets import interact, FloatSlider
from IPython.display import display

# ---------- 轨迹生成 ----------
T = 20          # ms
dt = 1e-5       # ms
t = np.arange(0, T, dt)
n = len(t)

#-------------------光泵浦调制---------------------------#
duration_op = 0.1            # ms
amplitude_op = 10            # kHz
frequency_op = 0.1           # kHz
duty_op = duration_op * frequency_op
Rop = amplitude_op * signal.square(2 * np.pi * frequency_op * t, duty=duty_op) + amplitude_op
#******************************************************#

#-------------------pi 脉冲调制---------------------------#
frequency_pi = 10            # kHz
amplitude_pi = 500           # kHz
duration_pi = np.pi / (2 * amplitude_pi)  # ms
duty_pi = duration_pi * frequency_pi
theta_pi = np.pi / 180 * 0.  # radians
phi_pi = 0
pi_envelope = amplitude_pi * signal.square(2 * np.pi * frequency_pi * t, duty=duty_pi) + amplitude_pi
omega_pix = pi_envelope * np.sin(theta_pi) * np.cos(phi_pi)
omega_piy = pi_envelope * np.sin(theta_pi) * np.sin(phi_pi)
omega_piz = pi_envelope * np.cos(theta_pi)
#******************************************************#

#-------------------偏置磁场---------------------------#
omega_x0=0.1
omega_y0=0.1
omega_z0=0.5
#******************************************************#

Gamma = 0.01
omega_x = omega_x0 + 1 * omega_pix
omega_y = omega_y0 + 1 * omega_piy
omega_z = omega_z0 + 1 * omega_piz

Pxarray = np.zeros(n)
Pyarray = np.zeros(n)
Pzarray = np.zeros(n)
Px, Py, Pz = 0, 0, 0

for i in trange(n):
    Pxarray[i] = Px
    Pyarray[i] = Py
    Pzarray[i] = Pz
    Px += (omega_y[i]*Pz - omega_z[i]*Py - Px*Gamma - Rop[i]*Px) * dt
    Py += (-omega_x[i]*Pz + omega_z[i]*Px - Py*Gamma - Rop[i]*Py) * dt
    Pz += (omega_x[i]*Py - omega_y[i]*Px - Pz*Gamma - Rop[i]*Pz + Rop[i]) * dt

# ---------- 缩减轨迹点数 ----------
skip = 400
Pxarray = Pxarray[::skip]
Pyarray = Pyarray[::skip]
Pzarray = Pzarray[::skip]
trajectory = np.vstack((Pxarray, Pyarray, Pzarray)).T

# ---------- 动画绘制函数 ----------
def plot_bloch_animation(elev=20, azim=45):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Bloch 球
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, edgecolor='gray')

    # 坐标轴
    arrow_len = 1
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='k', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='k', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='k', arrow_length_ratio=0.05)
    ax.text(1.3, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.3, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.3, 'Z', fontsize=12)

    # 添加主平面
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.08, color='red')     # XY 平面（Z=0）
    ax.plot_surface(zz, yy, xx, alpha=0.08, color='green')   # YZ 平面（X=0）
    ax.plot_surface(xx, zz, yy, alpha=0.08, color='blue')    # XZ 平面（Y=0）

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Bloch Vector with XY Projection and Coordinate Planes", fontsize=14)
    ax.view_init(elev=elev, azim=azim)

    # 初始化动画元素
    vector = [ax.quiver(0, 0, 0, trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                        color='red', linewidth=3, arrow_length_ratio=0.1)]
    projection = [ax.quiver(0, 0, 0, trajectory[0, 0], trajectory[0, 1], 0,
                            color='green', linewidth=2, arrow_length_ratio=0.1)]

    def update(i):
        m = trajectory[i]

        # 更新 Bloch 矢量
        vector[0].remove()
        vector[0] = ax.quiver(0, 0, 0, m[0], m[1], m[2],
                              color='red', linewidth=3, arrow_length_ratio=0.1)

        # 更新 XY 平面投影
        projection[0].remove()
        projection[0] = ax.quiver(0, 0, 0, m[0], m[1], 0,
                                  color='green', linewidth=2, arrow_length_ratio=0.1)

        return vector[0], projection[0]

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=20, blit=False)
    plt.tight_layout()
    plt.show()

# ---------- 添加交互式滑块 ----------
interact(
    plot_bloch_animation,
    elev=FloatSlider(min=0, max=90, step=1, value=20, description='Elevation'),
    azim=FloatSlider(min=0, max=360, step=1, value=45, description='Azimuth')
);