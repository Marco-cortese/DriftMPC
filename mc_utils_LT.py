# Equivalent script to InitParam.m, but for Python
################################################################################

# imports
import numpy as np
# useful functions from numpy (code more readable for matlab users)
π = 3.14159265358979323846264338327950288419716939937510582
np.random.seed(42)
def cot(x): return 1/np.tan(x) # cotangent function
np.set_printoptions(precision=6, formatter={'float': '{:+.6f}'.format}) # for better readability with sign
from scipy.io import loadmat # importing loadmat to read .mat files
from tqdm import tqdm # for progress bar
import matplotlib.pyplot as plt # for plotting
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

################################################################################
# from Fufy.mat
l = 0.56 # [m] wheelbase
# t = 0.49 # [m] car width
t = 0.41 # [m] car width
R_f, R_r = 0.095, 0.095 # [m] front and rear wheel radius
g = 9.81 # [m/s^2] gravity acceleration
steering_ratio_f = 0.6 # [-] steering ratio front

################################################################################
m        = 9.4+9.4 # [kg] massa veicolo con mini PC e tutto il resto
b        = l/2 # rear axle distance to CoG
a        = l-b # front axle distance to CoG
J_CoG    = 9.4*a**2+9.4*b**2 # Inertia axis z[kg*m^2]
J_CoG_real = J_CoG*0.9
Fz_Front_ST = 9.4*g # [N]
Fz_Rear_ST  = 9.4*g # [N]
Fz_Tot   = Fz_Rear_ST + Fz_Front_ST # [N] 
# Second test: height of CoG
H = 160/1000 # [m] 
h = (Fz_Rear_ST*l/(Fz_Tot)-(l-b))/np.tan(np.asin(H/l))+(R_r+R_f)/2 # [m] new CoG height
Fz_Front_nominal = Fz_Front_ST # for simulink 
Fz_Rear_nominal = Fz_Rear_ST # for simulink

#_____ Tires 
R_fl                = R_f # [m] front left wheel radius
R_fr                = R_f # [m] front right wheel radius
R_rl                = R_r # [m] rear left wheel radius
R_rr                = R_r # [m] rear right wheel radius

#_____ starting position _ 
x0                   = 0 # [m] distance CoG to y-axis (X0)
y0                   = 0 # [m] distance CoG to x-axis (Y0)
gamma                = π/2 #[rad] starting angle respect to x-axis (gamma0)


#___ Kinematic Condition
Toe_fl              = -0*π/180
Toe_fr              = -Toe_fl
Toe_rl              = 0*π/180
Toe_rr              = -Toe_rl 

t_front             = t # [m] track width front
t_rear              = t # [m] track width rear

μf, μr = 0.8, 0.8   # [] friction coefficients front and rear
Cyf = 370.36;  # [N/rad] Cornering stiffness front tyre
Cyr = 1134.05; # [N/rad] Cornering stiffness rear tyre

# constraints
MAX_DELTA = 25 * π / 180  # [rad] maximum steering angle in radians
MAX_V, MIN_V = 10, 0.1 # [m/s] maximum velocity
# MAX_FX = 0.8 * μr*Fz_Rear # [N] maximum rear longitudinal force
# MAX_FX, MIN_FX = 30, 0.0 # [N] maximum rear longitudinal force
MAX_FX, MIN_FX = 0.9 * μr * Fz_Rear_ST, 0.0 # [N] maximum rear longitudinal force
# MAX_FX, MIN_FX = μr * Fz_Rear - 5, 0.0 # [N] maximum rear longitudinal force

####################################################################################################
# tire model
# μf, μr = 0.8, 0.8   # [] friction coefficients front and rear
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
# def tire(α, Fx, Fz, μ, Cy): return fiala_tanh_np(α, Fx, Fz, μ, Cy) # choose the tire model (fiala or fiala_tanh)
def tire(α, Fx, Fz, μ, Cy): return fiala_np(α, Fx, Fz, μ, Cy) # choose the tire model (fiala or fiala_tanh)

# useful functions
def f_αf(δ, v, β, r): return δ - np.arctan2(v*np.sin(β) + a*r, v*np.cos(β)) # front slip angle function
def f_αr(δ, v, β, r): return -np.arctan2(v*np.sin(β) - b*r, v*np.cos(β)) # rear slip angle function

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

def car_anim(xs, us, dt, ic=(0.0,0.0,0.0), follow=False, fps=60.0, speed=1.0, title='Car Animation', get_video=False, static_img=False):
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display

    assert xs.shape[-1] == 3, "Input must be a 3-element array [V, β, r]"
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
        rl = size_mult*np.array([-b, +t/2]) # rear left
        rr = size_mult*np.array([-b, -t/2]) # rear right
        fl = size_mult*np.array([+a, +t/2]) # front left
        fr = size_mult*np.array([+a, -t/2]) # front right
        
        # Create a polygon for the car shape
        car_corners_lf = np.array([rl, rr, fr, fl, rl]).reshape(5, 2)  # Closed shape for the car

        # Transform the car shape to the global frame
        car_corners_gf = car_corners_lf @ r1.T + p  # rototranslation to global frame

        # draw the 4 wheels as 4 rectangles
        wheel_w, wheel_h = 0.1, 0.2  # wheel width and height
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

        plt.close(fig)  # Close the figure to avoid displaying it in Jupyter Notebook

        # anim.save('car_animation.gif', fps=FPS, dpi=50)  # save animation as gif
        # anim.save('car_animation.mp4', fps=fps, extra_args=['-vcodec', 'libx264']) # save animation as mp4

        if get_video: return display(HTML(anim.to_html5_video()))
        else: return display(HTML(anim.to_jshtml()))