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

# ---------- 光泵浦调制 ----------
duration_op = 0.1            # ms
amplitude_op = 10            # kHz
frequency_op = 0.1           # kHz
duty_op = duration_op * frequency_op
Rop = amplitude_op * signal.square(2 * np.pi * frequency_op * t, duty=duty_op) + amplitude_op

# ---------- π 脉冲调制 ----------
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

# ---------- 偏置磁场 ----------
omega_x0 = 0.1
omega_y0 = 0.1
omega_z0 = 0.5

Gamma = 0.01
omega_x = omega_x0 + omega_pix
omega_y = omega_y0 + omega_piy
omega_z = omega_z0 + omega_piz

# ---------- 积分演化 ----------
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

    # 坐标轴箭头
    arrow_len = 1
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='k', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='k', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='k', arrow_length_ratio=0.05)
    ax.text(1.3, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.3, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.3, 'Z', fontsize=12)

    # 主平面
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.08, color='red')     # XY 平面
    ax.plot_surface(zz, yy, xx, alpha=0.08, color='green')   # YZ 平面
    ax.plot_surface(xx, zz, yy, alpha=0.08, color='blue')    # XZ 平面

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=10)

    ax.set_title("Bloch Vector with π Pulse, Bias Field and Optical Pumping", fontsize=14)
    ax.view_init(elev=elev, azim=azim)

    # 偏置磁场箭头
    ax.quiver(0, 0, 0, omega_x0, omega_y0, omega_z0, color='orange', linewidth=2, arrow_length_ratio=0.1)
    ax.text(omega_x0*1.2, omega_y0*1.2, omega_z0*1.2, 'Bias B', color='orange', fontsize=10)

    # π 脉冲方向箭头（加粗）
    pi_x = np.sin(theta_pi) * np.cos(phi_pi)
    pi_y = np.sin(theta_pi) * np.sin(phi_pi)
    pi_z = np.cos(theta_pi)
    ax.quiver(0, 0, 0, pi_x, pi_y, pi_z, color='blue', linewidth=2, arrow_length_ratio=0.1)
    ax.text(pi_x*0.8 + 0.4, pi_y*0.8, pi_z*0.8, 'π Pulse B', color='blue', fontsize=10)

    # 光泵浦箭头（沿 z，注释偏移）
    ax.quiver(0, 0, 0, 0, 0, 1, color='magenta', linewidth=2, arrow_length_ratio=0.1)
    ax.text(0.1, 0, 1.2, 'Optical Pumping', color='magenta', fontsize=10)

    # 动画矢量初始化
    vector = [ax.quiver(0, 0, 0, trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                        color='red', linewidth=3, arrow_length_ratio=0.1)]
    projection = [ax.quiver(0, 0, 0, trajectory[0, 0], trajectory[0, 1], 0,
                            color='green', linewidth=2, arrow_length_ratio=0.1)]

    # 2D 标签：光泵状态 & Bloch 坐标
    pump_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='magenta')
    coord_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, fontsize=12, color='black')
    time_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, fontsize=12, color='gray')

    def update(i):
        m = trajectory[i]
        vector[0].remove()
        vector[0] = ax.quiver(0, 0, 0, m[0], m[1], m[2],
                            color='red', linewidth=3, arrow_length_ratio=0.1)
        projection[0].remove()
        projection[0] = ax.quiver(0, 0, 0, 10*m[0], 10*m[1], 0,
                                color='green', linewidth=2, arrow_length_ratio=0.1)

        idx = i * skip
        pump_state = "ON" if Rop[idx] > amplitude_op else "OFF"
        pump_text.set_text(f"Optical Pump: {pump_state}")
        coord_text.set_text(f"Bloch Vector = ({m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f})")
        time_ms = idx * dt
        time_text.set_text(f"Time = {time_ms:.2f} ms")

        return vector[0], projection[0], pump_text, coord_text, time_text


    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=20, blit=False)
    plt.tight_layout()
    plt.show()

# ---------- 交互滑块 ----------
interact(
    plot_bloch_animation,
    elev=FloatSlider(min=0, max=90, step=1, value=20, description='Elevation'),
    azim=FloatSlider(min=0, max=360, step=1, value=45, description='Azimuth')
)
