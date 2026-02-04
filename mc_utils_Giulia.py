# Equivalent script to InitParam.m, but for Python
################################################################################

# imports
import numpy as np
import scipy.io as sio
# useful functions from numpy (code more readable for matlab users)
π = 3.14159265358979323846264338327950288419716939937510582
np.random.seed(42)
def cot(x): return 1/np.tan(x) # cotangent function
np.set_printoptions(precision=6, formatter={'float': '{:+.6f}'.format}) # for better readability with sign
from scipy.io import loadmat # importing loadmat to read .mat files
from tqdm import tqdm # for progress bar
import matplotlib.pyplot as plt # for plotting
# casadi/acados model
import casadi as ca
from casadi import SX, sqrt, atan, tan, sin, cos, tanh, atan2, fabs, vertcat, if_else, sign
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
# suppress warnings acados
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='acados_template')
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
# vehicle parameters
g = 9.81 # m/s^2 (gravity)
a = 1.315 # m (front axle distance from COG)
b = 1.505 # m (rear axle distance from COG)
w = a+b # m (wheelbase)
h = 0.592 # m (COG height from ground)
tf = 1.557/2 # m (front half track)
tr = 1.625/2 # m (rear half track)
cw = (tr+tf)/2 # m (average track)
k_roll_f = 126254.887920000 # front roll stiffness Nm/rad
k_roll_r = 68944.0781250000 # rear roll stiffness Nm/rad
d = 0.0619840425531915 # roll center height
df = 0.041000000000000 # front roll center height
dr = 0.086000000000000 # rear roll center height
k_roll_tot = k_roll_f + k_roll_r
SR = 1/12.3 # steering ratio
m = 1.740132e+03 # kg (vehicle total mass)
Iz = 2129.335721 # kgm^2
Fz_Front_ST = m*9.81*b/(a+b)
Fz_Rear_ST = m*9.81*a/(a+b)
MF_front = sio.loadmat("MF_battipaglia_mergedDTM.mat")['MF_front']
MF_rear = sio.loadmat("MF_battipaglia_mergedDTM.mat")['MF_rear']
MF = np.stack([MF_front, MF_rear], 1).reshape([2,5])

# constraints
MAX_DELTA = SR*(540) * π / 180  # [rad] maximum steering angle in radians
MAX_V, MIN_V = 100/3.6, 0.1 # [m/s] maximum velocity
# MAX_FX = 0.8 * μr*Fz_Rear # [N] maximum rear longitudinal force
# MAX_FX, MIN_FX = 30, 0.0 # [N] maximum rear longitudinal force
μr = 1.0 # rear friction coefficient
MAX_FX, MIN_FX = 1.2 * μr * Fz_Rear_ST, -1.2 * μr * Fz_Rear_ST # [N] maximum rear longitudinal force
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

def F_mf_Fz_front(coeff, alpha, Fz):

    #Pacejka Magic Formula 5.2 with nominal load contribution
    B = coeff[0]
    C = coeff[1]
    D = (coeff[2]*Fz + coeff[3])*Fz
    E = coeff[4]
    F = D*sin(C*atan(B*alpha - E*(B*alpha - atan(B*alpha))))
    return F

def F_mf_Fz_rear(coeff, alpha, Fz, Fx):

    #Pacejka Magic Formula 5.2 with nominal load contribution
    B = coeff[0]
    C = coeff[1]
    D = (coeff[2]*Fz + coeff[3])*Fz
    D = if_else(D**2 < Fx**2, 0.001, sqrt(D**2 - Fx**2))  # combined model
    E = coeff[4]
    F = D*sin(C*atan(B*alpha - E*(B*alpha - atan(B*alpha))))
    return F
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
#--------------------------------------  MODELS    ---------------------------------------------------#


def DTM_model_LT_TOT(Ts):
    # variables
    v = SX.sym('v') # velocity
    beta = SX.sym('beta') # sideslip angle
    r = SX.sym('r') # yaw rate
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    dFz_x = SX.sym('dFz_x') # longitudinal load transfer
    dFz_yf = SX.sym('dFz_yf') # lateral load transfer front
    dFz_yr = SX.sym('dFz_yr') # lateral load transfer rear
    x = vertcat(v, beta, r, delta, Fx, dFz_x, dFz_yf, dFz_yr) # state vector

    d_delta = SX.sym('d_delta') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force
    u = vertcat(d_delta, d_Fx) # u input vector 

    # State derivatives (x_dot)
    v_dot = SX.sym('v_dot')
    beta_dot = SX.sym('beta_dot')
    r_dot = SX.sym('r_dot')
    delta_dot = SX.sym('delta_dot')
    Fx_dot = SX.sym('Fx_dot')
    dFz_x_dot = SX.sym('dFz_x_dot')
    dFz_yf_dot = SX.sym('dFz_yf_dot')
    dFz_yr_dot = SX.sym('dFz_yr_dot')
    x_dot = vertcat(v_dot, beta_dot, r_dot, delta_dot, Fx_dot, dFz_x_dot, dFz_yf_dot, dFz_yr_dot) # state dot vector
    
    # tire model
    alpha_fl = delta - atan2(v*sin(beta) + a*r, v*cos(beta) - r*cw/2)
    alpha_fr = delta - atan2(v*sin(beta) + a*r, v*cos(beta) + r*cw/2)
    alpha_rl = - atan2(v*sin(beta) - a*r, v*cos(beta) - r*cw/2)
    alpha_rr = - atan2(v*sin(beta) - a*r, v*cos(beta) + r*cw/2)

    # choose the tire model
    # def tire(alpha, Fx, Fz, μ, Cy): return fiala_tanh_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return pacejka(α, Fx, Fz, μ, Cy) # choose the tire model


    Fz_FL = Fz_Front_ST - dFz_x/2 - dFz_yf/2
    Fz_FR = Fz_Front_ST - dFz_x/2 + dFz_yf/2
    Fz_RL = Fz_Rear_ST + dFz_x/2 - dFz_yr/2
    Fz_RR = Fz_Rear_ST + dFz_x/2 + dFz_yr/2

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = F_mf_Fz_front(MF[0,:], alpha_fl, Fz_FL)
    Fy_fr = F_mf_Fz_front(MF[0,:], alpha_fr, Fz_FR)
    Fy_rl = F_mf_Fz_rear(MF[1,:], alpha_rl, Fz_RL, Fxl)
    Fy_rr = F_mf_Fz_rear(MF[1,:], alpha_rr, Fz_RR, Fxr)

    LT_x = (Fx - (Fy_fl + Fy_fr)*sin(delta))*h/w # longitudinal load transfer

    sum_Fy = Fy_rl + Fy_rr +Fy_fl*cos(delta) + Fy_fr*cos(delta)
    LT_yf = sum_Fy*((k_roll_f * (h-d)/(k_roll_tot*2*tf) + b*df/(w*2*tf)))
    LT_yr = sum_Fy*(k_roll_r * (h-d)/(k_roll_tot*2*tr) + a*dr/(w*2*tr))


    # symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(delta-beta) + Fx*cos(beta) + (Fy_rl + Fy_rr)*sin(beta)) / m # V dot
    dt_beta = (+(Fy_fl + Fy_fr)*cos(delta-beta) - Fx*sin(beta) + (Fy_rl + Fy_rr)*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(delta) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*cw/2) / Iz # r dot

    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force
    dt_dFz_x = (LT_x - dFz_x)/Ts # longitudinal load transfer dynamics
    dt_dFz_yf = (LT_yf - dFz_yf)/Ts # lateral load transfer front dynamics
    dt_dFz_yr = (LT_yr - dFz_yr)/Ts # lateral load transfer rear dynamics

    
    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx, dt_dFz_x, dt_dFz_yf, dt_dFz_yr) # equations of motion


    # create the model
    model = AcadosModel()
    model.name='dtm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model


def STM_model_dt_inputs():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, beta, r, delta, Fx) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

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

    # choose the tire model
    def tire(alpha, Fx, Fz, μ, Cy): return fiala_tanh_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front_ST, μf, Cyf) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear_ST, μr, Cyr) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / J_CoG # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_LT_dt_inputs():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    dFz = SX.sym('dFz') # load transfer

    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force


    x = vertcat(v, beta, r, delta, Fx, dFz) # state vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

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
    
    # choose the tire model
    def tire(alpha, Fx, Fz, μ, Cy): return fiala_tanh_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front_ST, μf, Cyf) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear_ST, μr, Cyr) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / J_CoG # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx, 0) # equations of motion (no LT -> d_Fz_dot = 0 Fz static)

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def DTM_model_dt_inputs_sim():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force


    x = vertcat(v, beta, r, delta, Fx) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_fl = delta - atan2(v*sin(beta) + a*r, v*cos(beta) - r*cw/2)
    alpha_fr = delta - atan2(v*sin(beta) + a*r, v*cos(beta) + r*cw/2)
    alpha_rl = - atan2(v*sin(beta) - a*r, v*cos(beta) - r*cw/2)
    alpha_rr = - atan2(v*sin(beta) - a*r, v*cos(beta) + r*cw/2)

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

    # choose the tire model
    # def tire(alpha, Fx, Fz, μ, Cy): return fiala_tanh_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model
    def tire(α, Fx, Fz, μ, Cy): return pacejka(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = tire(alpha_fl, 0.0, Fz_Front_ST/2, μf-0.05, (Cyf/2)*1.05) # lateral force front-left
    Fy_fr = tire(alpha_fr, 0.0, Fz_Front_ST/2, μf-0.05, (Cyf/2)*1.05) # lateral force front-right
    Fy_rl = tire(alpha_rl, Fxl, Fz_Rear_ST/2, μr-0.05, (Cyr/2)*1.05) # lateral force rear
    Fy_rr = tire(alpha_rr, Fxr, Fz_Rear_ST/2, μr-0.05, (Cyr/2)*1.05) # lateral force rear


    # define the symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(delta-beta) + Fx*cos(beta) + (Fy_rl + Fy_rr)*sin(beta)) / m # V dot
    dt_beta = (+(Fy_fl + Fy_fr)*cos(delta-beta) - Fx*sin(beta) + (Fy_rl + Fy_rr)*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(delta) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*cw/2) / J_CoG_real # r dot

    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def DTM_model_LT_dt_inputs_sim(Ts):
    # variables
    v = SX.sym('v') # velocity
    beta = SX.sym('beta') # sideslip angle
    r = SX.sym('r') # yaw rate
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force
    dFz = SX.sym('dFz')
    x = vertcat(v, beta, r, delta, Fx, dFz)

    d_delta = SX.sym('d_delta') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force
    u = vertcat(d_delta, d_Fx) # u input vector 

    # State derivatives (x_dot)
    v_dot = SX.sym('v_dot')
    beta_dot = SX.sym('beta_dot')
    r_dot = SX.sym('r_dot')
    delta_dot = SX.sym('delta_dot')
    Fx_dot = SX.sym('Fx_dot')
    dFz_dot = SX.sym('dFz_dot')
    x_dot = vertcat(v_dot, beta_dot, r_dot, delta_dot, Fx_dot, dFz_dot)
    
    # tire model
    alpha_fl = delta - atan2(v*sin(beta) + a*r, v*cos(beta) - r*cw/2)
    alpha_fr = delta - atan2(v*sin(beta) + a*r, v*cos(beta) + r*cw/2)
    alpha_rl = - atan2(v*sin(beta) - a*r, v*cos(beta) - r*cw/2)
    alpha_rr = - atan2(v*sin(beta) - a*r, v*cos(beta) + r*cw/2)

    # choose the tire model
    # def tire(alpha, Fx, Fz, μ, Cy): return fiala_tanh_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model
    def tire(α, Fx, Fz, μ, Cy): return pacejka(α, Fx, Fz, μ, Cy) # choose the tire model

    Fz_Front = Fz_Front_ST - dFz
    Fz_Rear = Fz_Rear_ST + dFz

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = tire(alpha_fl, 0.0, Fz_Front/2, μf, (Cyf/2)) # lateral force front-left
    Fy_fr = tire(alpha_fr, 0.0, Fz_Front/2, μf, (Cyf/2)) # lateral force front-right
    Fy_rl = tire(alpha_rl, Fxl, Fz_Rear/2, μr, (Cyr/2)) # lateral force rear
    Fy_rr = tire(alpha_rr, Fxr, Fz_Rear/2, μr, (Cyr/2)) # lateral force rear

    LT = (Fx - (Fy_fl + Fy_fr)*sin(delta))*h/l # total load transfer


    # define the symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(delta-beta) + Fx*cos(beta) + (Fy_rl + Fy_rr)*sin(beta)) / m # V dot
    dt_beta = (+(Fy_fl + Fy_fr)*cos(delta-beta) - Fx*sin(beta) + (Fy_rl + Fy_rr)*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(delta) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*cw/2) / J_CoG_real # r dot

    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force
    dt_dFz = (LT - dFz)/Ts
    
    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx, dt_dFz)


    # create the model
    model = AcadosModel()
    model.name='dtm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_dt_inputs_sim():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, beta, r, delta, Fx) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

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

    # choose the tire model
    def tire(alpha, Fx, Fz, μ, Cy): return fiala_ca(alpha, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front_ST, μf-0.05, Cyf*1.05) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear_ST, μr-0.05, Cyr*1.05) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / J_CoG # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_LT_sim():
    # State variables (x)
    v = SX.sym('v')       # velocity magnitude
    beta = SX.sym('beta') # sideslip angle
    r = SX.sym('r')       # yaw rate
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx')     # rear longitudinal force (treated as a state)
    dFz = SX.sym('dFz')
    x = vertcat(v, beta, r, delta, Fx, dFz)

    # State derivatives (x_dot)
    v_dot = SX.sym('v_dot')
    beta_dot = SX.sym('beta_dot')
    r_dot = SX.sym('r_dot')
    delta_dot = SX.sym('delta_dot')
    Fx_dot = SX.sym('Fx_dot')
    dFz_dot = SX.sym('dFz_dot')
    x_dot = vertcat(v_dot, beta_dot, r_dot, delta_dot, Fx_dot, dFz_dot)

    # Input variables (u)
    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx')       # change in rear longitudinal force
    u = vertcat(d_delta, d_Fx)

    # --- Tire Model ---
    def fiala_ca(alpha, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        Fy_lin = Cy * tan(alpha) - Cy**2 * fabs(tan(alpha)) * tan(alpha) / (3 * Fy_max) + Cy**3 * tan(alpha)**3 / (27 * Fy_max**2)
        Fy_sat = Fy_max * sign(alpha)
        return if_else(fabs(alpha) < atan(3*Fy_max/Cy), Fy_lin, Fy_sat)
    def pacejka(α, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        pB, pC, pD, pE = 4,1.8,Fy_max,-1 # Pacejka parameters
        return pD * sin(pC * atan(pB * α - pE * (pB*α - atan(pB*α))))

    tire = pacejka # Choose the tire model

    # Slip angles
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

    Fz_Front = Fz_Front_ST - dFz
    Fz_Rear = Fz_Rear_ST + dFz

    # Tire forces (using algebraic Fzf, Fzr)
    Fyf = tire(alpha_f, 0.0, Fz_Front, μf-0.05, Cyf*1.05)
    Fyr = tire(alpha_r, Fx, Fz_Rear, μr-0.05, Cyr*1.05)
    Fxr = Fx # Rear longitudinal force is a state

    ax = (Fx - Fyf*sin(delta))/m
    LT = m*ax*h/l # total load transfer
    
    # system dynamics
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / J_CoG # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force
    dt_dFz = (LT - dFz)/Ts
    
    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx, dt_dFz)
    
    # Combine into the final implicit model expression
    f_impl = x_dot - dx

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = f_impl
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.z = z # algebraic vector
    model.xdot = x_dot  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    β = SX.sym('β') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    x = vertcat(v, β, r) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(δ, Fx) # u input vector 

    # tire model
    αf = δ - atan2(v*sin(β) + a*r, v*cos(β))
    αr = -atan2(v*sin(β) - b*r, v*cos(β))
    def fiala_tanh_ca(α, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        αs = atan(Fy_max/Cy) # maximum slip angle
        return Fy_max * tanh(α / αs) # tanh approximation
    def fiala_ca(α, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        Fy_lin = Cy * tan(α) - Cy**2 * fabs(tan(α)) * tan(α) / (3 * Fy_max) + Cy**3 * tan(α)**3 / (27 * Fy_max**2)
        Fy_sat = Fy_max * sign(α)
        return if_else(fabs(α) < atan(3*Fy_max/Cy), Fy_lin, Fy_sat)

    # choose the tire model
    def tire(α, Fx, Fz, μ, Cy): return fiala_tanh_ca(α, Fx, Fz, μ, Cy) # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    Fyf = tire(αf, 0.0, Fz_Front, μf, Cyf) # lateral force front
    Fyr = tire(αr, Fx, Fz_Rear, μr, Cyr) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dv = (-Fyf*sin(δ-β) + Fxr*cos(β) + Fyr*sin(β)) / m # V dot
    dβ = (+Fyf*cos(δ-β) - Fxr*sin(β) + Fyr*cos(β)) / (m*v) - r # β dot
    dr = (a*Fyf*cos(δ) - b*Fyr) / J_CoG # r dot

    dx = vertcat(dv, dβ, dr) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_sim():
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    β = SX.sym('β') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    δ = SX.sym('δ') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    x = vertcat(v, β, r) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(δ, Fx) # u input vector 

    # tire model
    αf = δ - atan2(v*sin(β) + a*r, v*cos(β))
    αr = -atan2(v*sin(β) - b*r, v*cos(β))
    def fiala_tanh_ca(α, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        αs = atan(Fy_max/Cy) # maximum slip angle
        return Fy_max * tanh(α / αs) # tanh approximation
    def fiala_ca(α, Fx, Fz, μ, Cy):
        Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
        Fy_lin = Cy * tan(α) - Cy**2 * fabs(tan(α)) * tan(α) / (3 * Fy_max) + Cy**3 * tan(α)**3 / (27 * Fy_max**2)
        Fy_sat = Fy_max * sign(α)
        return if_else(fabs(α) < atan(3*Fy_max/Cy), Fy_lin, Fy_sat)

    # choose the tire model
    # def tire(α, Fx, Fz, μ, Cy): return fiala_tanh_ca(α, Fx, Fz, μ, Cy) # choose the tire model
    def tire(α, Fx, Fz, μ, Cy): return fiala_ca(α, Fx, Fz, μ, Cy) # choose the tire model

    # lateral forces
    # Fyf = tire(αf, 0.0, Fz_Front, μf, Cyf) # lateral force front
    Fyf = tire(αf, 0.0, Fz_Front, μf+0.05, Cyf) # lateral force front
    Fyr = tire(αr, Fx, Fz_Rear, μr+0.05, Cyr) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dv = (-Fyf*sin(δ-β) + Fxr*cos(β) + Fyr*sin(β)) / m # V dot
    dβ = (+Fyf*cos(δ-β) - Fxr*sin(β) + Fyr*cos(β)) / (m*v) - r # β dot
    dr = (a*Fyf*cos(δ) - b*Fyr) / J_CoG # r dot

    dx = vertcat(dv, dβ, dr) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model