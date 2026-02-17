# Equivalent script to InitParam.m, but for Python
################################################################################

# imports
import numpy as np
# useful functions from numpy (code more readable for matlab users)
π = 3.14159265358979323846264338327950288419716939937510582
# np.random.seed(42)
def cot(x): return 1/np.tan(x) # cotangent function
np.set_printoptions(precision=6, formatter={'float': '{:+.6f}'.format}) # for better readability with sign
from scipy.io import loadmat # importing loadmat to read .mat files
from scipy.linalg import block_diag # for block diagonal matrices
from tqdm import tqdm # for progress bar
import matplotlib.pyplot as plt # for plotting
# casadi/acados model
import casadi as ca
from casadi import SX, sqrt, atan, tan, sin, cos, tanh, atan2, fabs, vertcat, if_else, sign
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
# suppress warnings acados
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='acados_template')
# delete c_generated_code directory if it exists
import shutil
import os
if os.path.exists('c_generated_code'): shutil.rmtree('c_generated_code')


# Set plotting style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'monospace'
CM = 'jet' #'inferno'
plt.rcParams['image.cmap'] = CM

import os
home_dir = os.path.expanduser("~")
os.environ["ACADOS_SOURCE_DIR"] = f"{home_dir}/repos/acados"

print(f"ACADOS_SOURCE_DIR: {os.environ['ACADOS_SOURCE_DIR']}") # print the ACADOS source directory

g = 9.81 # [m/s^2] gravity acceleration

####### VI-GRADE CAR #########################################################################
l        = 2.74 # [m] wheelbase
m        = 2137.0 # [kg] massa veicolo con mini PC e tutto il resto
a        = 1.347 # front axle distance to CoG
b        = l-a # rear axle distance to CoG
cw       = 1.8 # [m] car width
J_CoG    = 1287.82 # Inertia axis z[kg*m^2]
J_CoG_real = J_CoG
Fz_Front_ST = m*g*b/l # [N]
Fz_Rear_ST  = m*g*a/l # [N]
Fz_Tot   = Fz_Rear_ST + Fz_Front_ST # [N] 
h = 0.328 # [m] CoG height
Fz_Front_nominal = Fz_Front_ST 
Fz_Rear_nominal = Fz_Rear_ST 
μf, μr = 1.0, 0.95   # [] friction coefficients front and rear
Cyf = 2*6.6e4;  # [N/rad] Cornering stiffness front tyre
Cyr = 2*7.9e4; # [N/rad] Cornering stiffness rear tyre
max_delta = 34*π/180 # [rad] maximum steering angle (at the ground) NOTE: change this
max_v, min_v = 35.0, 15.0 # [m/s] maximum and minimum velocity
max_d_fx = 1e5  # [N/s] maximum longitudinal force rate


# ####### MODEL CAR ################################################################################
# l = 0.56 # [m] wheelbase
# m        = 9.4+9.4 # [kg] massa veicolo con mini PC e tutto il resto
# b        = l/2 # rear axle distance to CoG
# a        = l-b # front axle distance to CoG
# cw        = 0.41 # [m] car width
# J_CoG    = 9.4*a**2+9.4*b**2 # Inertia axis z[kg*m^2]
# J_CoG_real = J_CoG*0.9
# Fz_Front_ST = 9.4*g # [N]
# Fz_Rear_ST  = 9.4*g # [N]
# Fz_Tot   = Fz_Rear_ST + Fz_Front_ST # [N] 
# h = (Fz_Rear_ST*l/(Fz_Tot)-(l-b))/np.tan(np.asin(0.16/l))+(0.095+0.095)/2 # [m] new CoG height
# Fz_Front_nominal = Fz_Front_ST # for simulink 
# Fz_Rear_nominal = Fz_Rear_ST # for simulink
# tf = 1.557/2 # m (front half track)
# tr = 1.625/2 # m (rear half track)
# k_roll_f = 126254.887920000 # front roll stiffness Nm/rad
# k_roll_r = 68944.0781250000 # rear roll stiffness Nm/rad
# d = 0.0619840425531915 # roll center height
# df = 0.041000000000000 # front roll center height
# dr = 0.086000000000000 # rear roll center height
# k_roll_tot = k_roll_f + k_roll_r
# μf, μr = 0.8, 0.8   # [] friction coefficients front and rear
# Cyf, Cyr = 370.36, 1134.05
# max_delta = 25*π/180 # [rad] maximum steering angle
# max_v, min_v = 10.0, 0.1 # [m/s] maximum and minimum velocity
# max_d_fx = 600.0  # [N/s] maximum longitudinal force rate



# constraints
MAX_DELTA = max_delta  # [rad] maximum steering angle in radians
MAX_V, MIN_V = max_v, min_v # [m/s] maximum velocity
# MAX_FX = 0.8 * μr*Fz_Rear # [N] maximum rear longitudinal force
# MAX_FX, MIN_FX = 30, 0.0 # [N] maximum rear longitudinal force
MAX_FX, MIN_FX = 0.9 * μr * Fz_Rear_ST, -0.9 * μr * Fz_Rear_ST # [N] maximum rear longitudinal force
# MAX_FX, MIN_FX = μr * Fz_Rear - 5, 0.0 # [N] maximum rear longitudinal force

MAX_BETA = π/2 - 1.1* MAX_DELTA # limit on sideslip angle to avoid spinning (testacoda) (NOTE: there should be a geometrical limit)

MAX_D_DELTA = 1.5*2*π  # [rad/s] maximum steering angle rate in radians
MAX_D_FX = max_d_fx  # [N/s] maximum longitudinal force rate

print(f"Car parameters: m={m} kg, J_CoG={J_CoG} kg*m^2, a={a} m, b={b} m, h={h:.3f} m")
print(f'MAX_DELTA: {np.rad2deg(MAX_DELTA):.2f} deg, MAX_V: {MAX_V} m/s, MIN_V: {MIN_V} m/s, MAX_FX: {MAX_FX} N, MIN_FX: {MIN_FX} N')

####################################################################################################
# tire model
def fiala_tanh_np(α, Fx, Fz, μ, Cy): # tanh fiala approximation
    assert Fx**2 <= μ**2 * Fz**2, "Longitudinal force exceeds maximum limit"
    Fy_max = np.sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    αs = np.atan(Fy_max/Cy) # maximum slip angle
    return Fy_max * np.tanh(α / αs) # tanh approximation
def fiala_np(α, Fx, Fz, μ, Cy):
    # assert Fx**2 <= μ**2 * Fz**2, f'Longitudinal force {Fx:.2f} exceeds maximum limit {μ**2 * Fz**2:.2f}'
    Fy_max = np.sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    Fy_lin = Cy * np.tan(α) - Cy**2 * np.abs(np.tan(α)) * np.tan(α) / (3 * Fy_max) + Cy**3 * np.tan(α)**3 / (27 * Fy_max**2)
    Fy_sat = Fy_max * np.sign(α)
    return np.where(np.abs(α) < np.atan(3*Fy_max/Cy), Fy_lin, Fy_sat)
def tire(α, Fx, Fz, μ, Cy): return fiala_tanh_np(α, Fx, Fz, μ, Cy) # choose the tire model (fiala or fiala_tanh)
# def tire(α, Fx, Fz, μ, Cy): return fiala_np(α, Fx, Fz, μ, Cy) # choose the tire model (fiala or fiala_tanh)

# useful functions
def f_αf(δ, v, β, r): return δ - np.arctan2(v*np.sin(β) + a*r, v*np.cos(β)) # front slip angle function
def f_αr(δ, v, β, r): return -np.arctan2(v*np.sin(β) - b*r, v*np.cos(β)) # rear slip angle function

# utils
def clear_previous_simulation():
    vars = globals().copy()
    for v in globals().copy():
        if(type(vars[v]) is AcadosSimSolver or type(vars[v]) is AcadosOcpSolver):
            del globals()[v]

def piecewise_constant(vals, durs, dt):
    if type(vals) is list: vals = np.array(vals).reshape(-1, 1)  # ensure vals is a column vector
    assert vals.shape[0] == len(durs), "vals and durs must have the same length"
    assert vals.ndim == 2, "vals must be a 2D array"
    tot_samples = int(round(np.sum(durs) / dt))
    durs_n = [int(round(d/dt)) for d in durs]
    pc = np.concatenate([np.full((n,vals.shape[1]), v) for v, n in zip(vals, durs_n)], axis=0)
    return np.append(pc, vals[-1:], axis=0), tot_samples * dt

def compute_num_steps(ts_sim, Ts, Tf):
    # check consistency
    assert abs(Ts/ts_sim - round(Ts/ts_sim)) < 1e-6, f"ts_sim {ts_sim} must be a divisor of Ts {Ts}"
    assert abs(Tf/ts_sim - round(Tf/ts_sim)) < 1e-6, f"ts_sim {ts_sim} must be a divisor of Tf {Tf}"
    N_steps    = int(round(Tf/ts_sim)) # compute the number of simulation steps
    N_steps_dt = int(round(Tf/Ts)) # compute the number of steps for the discrete-time part of the loop
    n_update   = int(round(Ts/ts_sim)) # number of simulation steps every which to update the discrete-time part of the loop 
    return (N_steps, N_steps_dt, n_update)

def print_red(*args, **kwargs):
    print("\033[91m", end="")  # ANSI escape code for red text
    print(*args, **kwargs)
    print("\033[0m", end="")   # ANSI escape code to reset text color

def print_yellow(*args, **kwargs):
    print("\033[93m", end="")  # ANSI escape code for yellow text
    print(*args, **kwargs)
    print("\033[0m", end="")   # ANSI escape code to reset text color

# state space model
def d_vβr(vβr, δ, Fx):  # -> vβr dot
    v, β, r = vβr # unpack the state vector
    # assert v >= 0, "Velocity must be non-negative" # ensure velocity is non-negative
    if v < 0.001: v = 0.001 # avoid division by zero
    Fyf = tire(f_αf(δ,v,β,r), 0.0, Fz_Front_ST, μf, Cyf) # lateral force front
    Fyr = tire(f_αr(δ,v,β,r), Fx, Fz_Rear_ST, μr, Cyr) # lateral force rear
    Fxr = Fx # rear longitudinal force
    return np.array([ # equations of motion
        (-Fyf*np.sin(δ-β) + Fxr*np.cos(β) + Fyr*np.sin(β)) / m, # V dot
        (+Fyf*np.cos(δ-β) - Fxr*np.sin(β) + Fyr*np.cos(β)) / (m*v) - r, # β dot
        (a*Fyf*np.cos(δ) - b*Fyr) / J_CoG # r dot
    ])

def stm_rk4(vβr, δ, Fx, dt=1e-3): # runge-kutta 4th order method
    k1 = d_vβr(vβr, δ, Fx) * dt
    k2 = d_vβr(vβr + k1/2, δ, Fx) * dt
    k3 = d_vβr(vβr + k2/2, δ, Fx) * dt
    k4 = d_vβr(vβr + k3, δ, Fx) * dt
    return vβr + (k1 + 2*k2 + 2*k3 + k4) / 6 # update the state vector

def sim_stm_fixed_u(vβr0, δ, Fx, sim_t=1, dt=1e-3, verbose=False): # simulate the STM
    # simulate the STM for sim_t seconds
    n_steps = int(sim_t/dt) # number of steps in the simulation
    # initialize the state vector
    state = np.zeros((n_steps, 3)) 
    state[0] = vβr0 # initial state in v,β,r format
    if verbose: print(f"Initial state: {state[0]} [v,β,r], δ={δ:.2f}, Fx={Fx:.2f}") # print the initial state in v,β,r format
    # run the simulation
    for i in range(1, n_steps):
        state[i] = stm_rk4(state[i-1], δ, Fx, dt) # update the state vector  
    if verbose: print(f"Final state:   {state[-1]} [v,β,r]") # print the final state in v,β,r format
    return state

def vel2beta(uvr): # -> vβr
    """Converts velocity components to sideslip angle and speed."""
    assert uvr.shape[-1] == 3, "Input must be a 3-element array [u, v, r]"
    uvr_shape = uvr.shape
    uvr = uvr.reshape(-1, 3)  # flatten the input to 2D if necessary
    u, v, r = uvr[:, 0], uvr[:, 1], uvr[:, 2]  # unpack velocity components
    V = np.sqrt(u**2 + v**2)  # speed
    β = np.arctan2(v, u)  # sideslip angle
    return np.stack([V, β, r], axis=-1).reshape(uvr_shape)  # reshape back to original shape if necessary

def beta2vel(vβr): # -> uvr
    """Converts sideslip angle and speed to velocity components."""
    assert vβr.shape[-1] == 3, "Input must be a 3-element array [V, β, r]"
    vβr_shape = vβr.shape
    vβr = vβr.reshape(-1, 3)  # flatten the input to 2D if necessary
    V, β, r = vβr[:, 0], vβr[:, 1], vβr[:, 2]  # unpack speed, sideslip angle, and yaw rate
    u = V * np.cos(β)  # longitudinal velocity component
    v = V * np.sin(β)  # lateral velocity component
    return np.stack([u, v, r], axis=-1).reshape(vβr_shape)  # reshape back to original shape if necessary

def car_anim(xs, us, dt, ic=(0.0,0.0,0.0), follow=False, fps=60.0, speed=1.0, title='Car Animation', get_video=False, static_img=False, in_notebook=True):
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display

    assert xs.shape[-1] == 3, "Input must be a 3-element array [V, β, r]"

    # put all nans and infs to zero
    if np.any(np.isnan(xs)) or np.any(np.isinf(xs)):
        print("Warning: NaNs or Infs found in state array, replacing with zeros.")
        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)


    vs, βs, rs = xs[:, 0], xs[:, 1], xs[:, 2] # unpack the vβr array
    δs, Fxs = us[:, 0], us[:, 1]  # unpack the control inputs
    drifts = 180/π*np.abs(βs + δs) # "amount of drift" as the sum of sideslip angle and steering angle 
    n = xs.shape[0]  # number of time steps
    # integrate the velocity components to get x, y, ψ
    xs, ys, ψs = np.zeros((n,)), np.zeros((n,)), np.zeros((n,))  # initialize arrays for x, y, ψ
    xs[0], ys[0], ψs[0] = ic  # initial conditions
    for i in range(1, n): # integrate the velocity components
        x, y, ψ, v, β, r = xs[i-1], ys[i-1], ψs[i-1], vs[i], βs[i], rs[i]  # unpack
        xs[i] = x + v*np.cos(ψ+β)*dt  
        ys[i] = y + v*np.sin(ψ+β)*dt
        ψs[i] = ψ + r*dt  

    def get_car_shapes(x, y, ψ, δ, size_mult=1.5): # -> car_corners_gf, wrl_gf, wrr_gf, wfl_gf, wfr_gf
        """Get the car shapes in the global frame."""
        p = np.array([x, y])  # position in the global frame
        r1 = np.array([[np.cos(ψ), -np.sin(ψ)], [np.sin(ψ),  np.cos(ψ)]]) # Rotation matrix for the yaw angle

        # create the car shape in the local frame
        rl = size_mult*np.array([-b, +cw/2]) # rear left
        rr = size_mult*np.array([-b, -cw/2]) # rear right
        fl = size_mult*np.array([+a, +cw/2]) # front left
        fr = size_mult*np.array([+a, -cw/2]) # front right
        
        # Create a polygon for the car shape
        car_corners_lf = np.array([rl, rr, fr, fl, rl]).reshape(5, 2)  # Closed shape for the car

        # Transform the car shape to the global frame
        car_corners_gf = car_corners_lf @ r1.T + p  # rototranslation to global frame

        # draw the 4 wheels as 4 rectangles
        wheel_w, wheel_h = 0.1, 0.2  # wheel width and height
        wheel_w, wheel_h = 0.18*l, 0.45*cw  # wheel width and height
        wheel = size_mult* np.array([  
                            [-wheel_h/2, -wheel_w/2],
                            [+wheel_h/2, -wheel_w/2],
                            [+wheel_h/2, +wheel_w/2],
                            [-wheel_h/2, +wheel_w/2],
                            [-wheel_h/2, -wheel_w/2]])  # Closed shape for the wheel
        
        r2 = np.array([[np.cos(δ), -np.sin(δ)], [np.sin(δ),  np.cos(δ)]]) # Rotation matrix for the steering angle
        wrl_lf = rl + wheel  # rear left wheel
        wrr_lf = rr + wheel  # rear right wheel
        wfl_lf = fl + wheel @ r2.T  # front left wheel
        wfr_lf = fr + wheel @ r2.T  # front right wheel
        
        # rototranslation to global frame
        wrl_gf = wrl_lf @ r1.T + p
        wrr_gf = wrr_lf @ r1.T + p  
        wfl_gf = wfl_lf @ r1.T + p
        wfr_gf = wfr_lf @ r1.T + p

        return car_corners_gf, wrl_gf, wrr_gf, wfl_gf, wfr_gf

    # create the figure and axis
    fig, ax = plt.subplots(1,2,figsize=(16, 9), width_ratios=[1, 6 if not static_img else 100])
    ax[1].set_aspect('equal')
    ax[1].set_xlim(np.min(xs)-1, np.max(xs)+1)
    ax[1].set_ylim(np.min(ys)-1, np.max(ys)+1)

    ax[1].set_xlabel('x [m]')
    ax[1].set_ylabel('y [m]')
    

    if not static_img: 
        slip = ax[1].scatter(xs, ys, c=np.abs(np.rad2deg(βs)), s=2, cmap=CM, vmin=0, vmax=30, alpha=0.6)
        cbar = plt.colorbar(slip, ax=ax[1], label='β [deg]')

        # in the 0 ax plot 2 bars with the Fx and δ values updating in time
        max_drift_angle = 45 # [deg] maximum drift angle for the bar plot
        bar = ax[0].bar([1,2,3], [Fxs[0], drifts[0]*MAX_FX/max_drift_angle, np.abs(δs[0]*MAX_FX/MAX_DELTA)], color='orange')
        ax0b = ax[0].twinx()  # create a twin axis
        ax[0].set_xticks([1, 2, 3])
        ax[0].set_xticklabels(['Fx', 'Drift', 'δ'])
        ax[0].set_ylim(0, MAX_FX)
        ax[0].grid(False), ax0b.grid(False)  # disable grid for the bar plot
        ax0b.set_ylim(0, MAX_DELTA * 180/π)  # set the limits of the twin axis
        
    fig.suptitle(title)

    plt.tight_layout()

    # create initial plots for the car and wheels
    car, wrl, wrr, wfl, wfr = get_car_shapes(xs[0], ys[0], ψs[0], δs[0])
    wheel_color, wheel_alpha = 'white', 1  # wheel color and alpha
    body_color, body_alpha = 'white', 0.7  # body color and alpha    
    car_plot, = ax[1].plot(car[:, 0], car[:, 1], color=body_color, alpha=body_alpha)
    wrl_plot, = ax[1].plot(wrl[:, 0], wrl[:, 1], color=wheel_color, alpha=wheel_alpha)
    wrr_plot, = ax[1].plot(wrr[:, 0], wrr[:, 1], color=wheel_color, alpha=wheel_alpha)
    wfl_plot, = ax[1].plot(wfl[:, 0], wfl[:, 1], color=wheel_color, alpha=wheel_alpha)
    wfr_plot, = ax[1].plot(wfr[:, 0], wfr[:, 1], color=wheel_color, alpha=wheel_alpha)

    k_static = 5.0 if static_img else 1.0  # speed factor for static images 

    xs = xs[::int(fps*speed*k_static)]
    ys = ys[::int(fps*speed*k_static)]
    ψs = ψs[::int(fps*speed*k_static)]
    Fxs = Fxs[::int(fps*speed*k_static)]
    δs = δs[::int(fps*speed*k_static)]
    drifts = drifts[::int(fps*speed*k_static)]

    if static_img: # plot all the frames as static images
        for i in tqdm(range(len(xs)), desc='Plotting frames', leave=False):
            car, wrl, wrr, wfl, wfr = get_car_shapes(xs[i], ys[i], ψs[i], δs[i])
            col = plt.get_cmap(CM)(i / len(xs))
            ax[1].plot(car[:, 0], car[:, 1], color=col, alpha=body_alpha)
            ax[1].plot(wrl[:, 0], wrl[:, 1], color=col, alpha=wheel_alpha)
            ax[1].plot(wrr[:, 0], wrr[:, 1], color=col, alpha=wheel_alpha)
            ax[1].plot(wfl[:, 0], wfl[:, 1], color=col, alpha=wheel_alpha)
            ax[1].plot(wfr[:, 0], wfr[:, 1], color=col, alpha=wheel_alpha)
        return fig

    else: # create the animation
        def update(frame):
            # Update the car and wheels positions    
            car, wrl, wrr, wfl, wfr = get_car_shapes(xs[frame], ys[frame], ψs[frame], δs[frame])
            # Update the plots
            car_plot.set_data(car[:, 0], car[:, 1])
            wrl_plot.set_data(wrl[:, 0], wrl[:, 1])
            wrr_plot.set_data(wrr[:, 0], wrr[:, 1])
            wfl_plot.set_data(wfl[:, 0], wfl[:, 1])
            wfr_plot.set_data(wfr[:, 0], wfr[:, 1])

            # update bar plot
            bar[0].set_height(Fxs[frame])
            bar[1].set_height(drifts[frame]*MAX_FX/max_drift_angle)
            bar[2].set_height(np.abs(δs[frame]*MAX_FX/MAX_DELTA))

            if follow:
                # set the limits of the axis
                window_size = 8.0 # [m] size of the window around the car
                ax[1].set_xlim(xs[frame] - window_size, xs[frame] + window_size)
                ax[1].set_ylim(ys[frame] - window_size, ys[frame] + window_size)

            return car_plot, wrl_plot, wrr_plot, wfl_plot, wfr_plot

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(xs), interval=1000/fps, blit=True)

        if in_notebook:
            plt.close(fig)  # Close the figure to avoid displaying it in Jupyter Notebook

            # anim.save('car_animation.gif', fps=FPS, dpi=50)  # save animation as gif
            # anim.save('car_animation.mp4', fps=fps, extra_args=['-vcodec', 'libx264']) # save animation as mp4

            if get_video: return display(HTML(anim.to_html5_video()))
            else: return display(HTML(anim.to_jshtml()))
        else: return anim

#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
#---------------------     TIRE MODELS    ------------------------------------------------------------#

    # tire model functions
def fiala_tanh_ca(alpha, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    alpha_s = atan(Fy_max/Cy) # maximum slip angle
    return Fy_max * tanh(alpha / alpha_s) # tanh approximation
def fiala_ca(alpha, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    Fy_lin = Cy * tan(alpha) - Cy**2 * fabs(tan(alpha)) * tan(alpha) / (3 * Fy_max) + Cy**3 * tan(alpha)**3 / (27 * Fy_max**2)
    Fy_sat = Fy_max * sign(alpha)
    return if_else(fabs(alpha) < atan(3*Fy_max/Cy), Fy_lin, Fy_sat)
def pacejka(α, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    pB, pC, pD, pE = 4,1.8,Fy_max,-1 # Pacejka parameters
    return pD * sin(pC * atan(pB * α - pE * (pB*α - atan(pB*α))))

#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
#--------------------------------------  MODELS    ---------------------------------------------------#


def STM_model_dt_inputs(Ts=0.01, tire=fiala_tanh_ca, μ_err=0.0, Cy_err=1.0):
    # variables
    v = SX.sym('v') # velocity
    β = SX.sym('β') # sideslip angle
    r = SX.sym('r') # yaw rate
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    d_δ = SX.sym('d_δ') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, β, r, δ, Fx) # state vector
    u = vertcat(d_δ, d_Fx) # u input vector 

    v_dot = SX.sym('v_dot') # velocity derivative
    β_dot = SX.sym('β_dot') # sideslip angle derivative
    r_dot = SX.sym('r_dot') # yaw rate derivative
    δ_dot = SX.sym('δ_dot') # wheel angle derivative
    Fx_dot = SX.sym('Fx_dot') # rear longitudinal force derivative
    x_dot = vertcat(v_dot, β_dot, r_dot, δ_dot, Fx_dot) # state derivative vector

    alpha_f = δ - atan2(v*sin(β) + a*r, v*cos(β))
    alpha_r = -atan2(v*sin(β) - b*r, v*cos(β))

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front_ST, μf+μ_err, Cyf*Cy_err) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear_ST, μr+μ_err, Cyr*Cy_err) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(δ-β) + Fxr*cos(β) + Fyr*sin(β)) / m # V dot
    dt_β = (+Fyf*cos(δ-β) - Fxr*sin(β) + Fyr*cos(β)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(δ) - b*Fyr) / J_CoG # r dot
    dt_delta = d_δ # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_β, dt_r, dt_delta, dt_Fx) # equations of motion

    model = AcadosModel()
    model.name='stm_model'
    model.f_expl_expr = dx 
    model.x = x  # state vector
    model.u = u  # u input vector
    model.f_impl_expr = x_dot - dx
    model.xdot = x_dot  # state derivative vector
    return model

def STM_model_dt_inputs_sim(): # kept for compatibility
    return STM_model_dt_inputs(Ts=0.01, tire=fiala_ca, μ_err=0.04, Cy_err=1.04)

def STM_model_LT_dt_inputs(Ts=0.01, tire=fiala_tanh_ca, μ_err=0.0, Cy_err=1.0):
    # variables
    v = SX.sym('v') # velocity
    β = SX.sym('β') # sideslip angle
    r = SX.sym('r') # yaw rate
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    dFz = SX.sym('dFz') # load transfer

    d_δ = SX.sym('d_δ') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, β, r, δ, Fx, dFz) # state vector
    u = vertcat(d_δ, d_Fx) # u input vector 

    # tire model
    alpha_f = δ - atan2(v*sin(β) + a*r, v*cos(β))
    alpha_r = -atan2(v*sin(β) - b*r, v*cos(β))

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front_ST, μf+μ_err, Cyf*Cy_err) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear_ST, μr+μ_err, Cyr*Cy_err) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(δ-β) + Fxr*cos(β) + Fyr*sin(β)) / m # V dot
    dt_β = (+Fyf*cos(δ-β) - Fxr*sin(β) + Fyr*cos(β)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(δ) - b*Fyr) / J_CoG # r dot
    dt_delta = d_δ # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_β, dt_r, dt_delta, dt_Fx, 0) # equations of motion (no LT -> d_Fz_dot = 0 Fz static)

    model = AcadosModel()
    model.name='stm_model'
    model.f_expl_expr = dx 
    model.x = x  # state vector
    model.u = u  # u input vector
    return model

def STM_model_LT_dt_inputs_sim(): # kept for compatibility
    return STM_model_LT_dt_inputs(Ts=0.01, tire=fiala_ca, μ_err=0.04, Cy_err=1.04)

def DTM_model_dt_inputs(Ts=0.01, tire=fiala_tanh_ca, μ_err=0.0, Cy_err=1.0):
    # variables
    v = SX.sym('v') # velocity
    β = SX.sym('β') # sideslip angle
    r = SX.sym('r') # yaw rate
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    d_δ = SX.sym('d_δ') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force

    x = vertcat(v, β, r, δ, Fx) # state vector
    u = vertcat(d_δ, d_Fx) # u input vector 

    alpha_fl = δ - atan2(v*sin(β) + a*r, v*cos(β) - r*cw/2)
    alpha_fr = δ - atan2(v*sin(β) + a*r, v*cos(β) + r*cw/2)
    alpha_rl = - atan2(v*sin(β) - a*r, v*cos(β) - r*cw/2)
    alpha_rr = - atan2(v*sin(β) - a*r, v*cos(β) + r*cw/2)

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = tire(alpha_fl, 0.0, Fz_Front_ST/2, μf-μ_err, (Cyf/2)*Cy_err) # lateral force front-left
    Fy_fr = tire(alpha_fr, 0.0, Fz_Front_ST/2, μf-μ_err, (Cyf/2)*Cy_err) # lateral force front-right
    Fy_rl = tire(alpha_rl, Fxl, Fz_Rear_ST/2, μr-μ_err, (Cyr/2)*Cy_err) # lateral force rear
    Fy_rr = tire(alpha_rr, Fxr, Fz_Rear_ST/2, μr-μ_err, (Cyr/2)*Cy_err) # lateral force rear

    # define the symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(δ-β) + Fx*cos(β) + (Fy_rl + Fy_rr)*sin(β)) / m # V dot
    dt_β = (+(Fy_fl + Fy_fr)*cos(δ-β) - Fx*sin(β) + (Fy_rl + Fy_rr)*cos(β)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(δ) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*cw/2) / J_CoG_real # r dot

    dt_delta = d_δ # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force

    dx = vertcat(dt_v, dt_β, dt_r, dt_delta, dt_Fx) # equations of motion

    model = AcadosModel()
    model.name='stm_model'
    model.f_expl_expr = dx 
    model.x = x  # state vector
    model.u = u  # u input vector
    return model

def DTM_model_dt_inputs_sim(): # kept for compatibility
    return DTM_model_dt_inputs(Ts=0.01, tire=fiala_ca, μ_err=0.04, Cy_err=1.04)

def DTM_model_LT_dt_inputs(Ts=0.01, tire=fiala_tanh_ca, μ_err=0.0, Cy_err=1.0):
    # variables
    v = SX.sym('v') # velocity
    β = SX.sym('β') # sideslip angle
    r = SX.sym('r') # yaw rate
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    dFz = SX.sym('dFz')
    x = vertcat(v, β, r, δ, Fx, dFz)

    d_δ = SX.sym('d_δ') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force
    u = vertcat(d_δ, d_Fx) # u input vector 

    # State derivatives (x_dot)
    v_dot = SX.sym('v_dot')
    β_dot = SX.sym('β_dot')
    r_dot = SX.sym('r_dot')
    δ_dot = SX.sym('δ_dot')
    Fx_dot = SX.sym('Fx_dot')
    dFz_dot = SX.sym('dFz_dot')
    x_dot = vertcat(v_dot, β_dot, r_dot, δ_dot, Fx_dot, dFz_dot)
    
    # tire model
    alpha_fl = δ - atan2(v*sin(β) + a*r, v*cos(β) - r*cw/2)
    alpha_fr = δ - atan2(v*sin(β) + a*r, v*cos(β) + r*cw/2)
    alpha_rl = - atan2(v*sin(β) - a*r, v*cos(β) - r*cw/2)
    alpha_rr = - atan2(v*sin(β) - a*r, v*cos(β) + r*cw/2)

    Fz_Front = Fz_Front_ST - dFz
    Fz_Rear = Fz_Rear_ST + dFz

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = tire(alpha_fl, 0.0, Fz_Front/2, μf-μ_err, (Cyf/2)*Cy_err) # lateral force front-left
    Fy_fr = tire(alpha_fr, 0.0, Fz_Front/2, μf-μ_err, (Cyf/2)*Cy_err) # lateral force front-right
    Fy_rl = tire(alpha_rl, Fxl, Fz_Rear/2, μr-μ_err, (Cyr/2)*Cy_err) # lateral force rear
    Fy_rr = tire(alpha_rr, Fxr, Fz_Rear/2, μr-μ_err, (Cyr/2)*Cy_err) # lateral force rear

    LT = (Fx - (Fy_fl + Fy_fr)*sin(δ))*h/l # total load transfer

    # define the symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(δ-β) + Fx*cos(β) + (Fy_rl + Fy_rr)*sin(β)) / m # V dot
    dt_β = (+(Fy_fl + Fy_fr)*cos(δ-β) - Fx*sin(β) + (Fy_rl + Fy_rr)*cos(β)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(δ) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*cw/2) / J_CoG_real # r dot

    dt_delta = d_δ # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force
    dt_dFz = (LT - dFz)/Ts
    
    dx = vertcat(dt_v, dt_β, dt_r, dt_delta, dt_Fx, dt_dFz)

    model = AcadosModel()
    model.name='dtm_model'
    model.f_expl_expr = dx 
    model.x = x  # state vector
    model.u = u  # u input vector
    return model

def DTM_model_LT_dt_inputs_sim(Ts): # kept for compatibility
    return DTM_model_LT_dt_inputs(Ts=Ts, tire=fiala_ca, μ_err=0.04, Cy_err=1.04)