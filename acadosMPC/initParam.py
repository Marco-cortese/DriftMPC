# Equivalent script to InitParam.m, but for Python
################################################################################

# imports
import numpy as np
# useful functions from numpy (code more readable for matlab users)
from numpy import pi as π, sqrt, cos, sin, exp, tan, arctan, atan, arctan2, atan2, arcsin, asin, abs, sign
def cot(x): return 1/tan(x) # cotangent function
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
plt.rcParams['image.cmap'] = 'inferno'

################################################################################
# from Fufy.mat
l = 0.56 # [m] wheelbase
t = 0.49 # [m] car width
R_f, R_r = 0.095, 0.095 # [m] front and rear wheel radius
g = 9.81 # [m/s^2] gravity acceleration
steering_ratio_f = 0.6 # [-] steering ratio front

mu_f, mu_r = 0.5, 0.47   # [] friction coefficients front and rear
Cyf = 370.36;  # [N/rad] Cornering stiffness front tyre
Cyr = 1134.05; # [N/rad] Cornering stiffness rear tyre

################################################################################
## CHANGING SECTION
# Manually update the library for the vehicle

m        = 9.4*2 # [kg] massa veicolo con mini PC e tutto il resto
b        = l/2 # rear axle distance to CoG
a        = l-b # front axle distance to CoG
J_CoG    = m/2*a**2+m/2*b**2 # Inertia axis z[kg*m^2]
Fz_Front = m/2*g # [N]
Fz_Rear  = m/2*g # [N]
Fz_Tot   = Fz_Rear + Fz_Front # [N] 
# Second test: height of CoG
H = 160/1000 # [m] 
h = (Fz_Rear*l/(Fz_Tot)-(l-b))*cot(asin(H/l))+(R_r+R_f)/2 # [m] new CoG height
Fz_Front_nominal = Fz_Front # for simulink 
Fz_Rear_nominal = Fz_Rear # for simulink

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



# useful functions
def diff_angle(a, b):
    """Returns the difference between two angles in radians, normalized to [-pi, pi]."""
    c1 = (a - b + π) % (2 * π) - π
    c2 = np.arctan2(np.sin(a - b), np.cos(a - b))
    assert np.allclose(c1, c2), f'{c1} != {c2}'
    return c1

assert a == l - b, "a must be equal to l - b"


def piecewise_constant(setpoints, setpoints_duration, Ts):

    '''
    Defines the sampled version with sample time Ts of a piecewise constant reference,
    
                 _
                | setpoints(0)      0   <= t < T_1
                | setpoints(1)      T_1 <= t < T_1 + T_2
      ref(t) = <
                | ...
                | setpoints(n-1)    T_1 + ... T_(n-1) <= t <= T_1 + ... + T_n
                 _
    
    where T_1, ..., T_n are the duration of each setpoint, i.e. the entries of setpoints_duration
    
    '''
    # compute the number of samples for each setpoint 
    n_samples = np.rint(setpoints_duration/Ts).astype(int)

    # compute the reference for each setpoint
    ref = setpoints[0]*np.ones((n_samples[0],))

    for i in range(1, len(setpoints)):
        ref = np.append(ref, setpoints[i]*np.ones((n_samples[i],)))

    # add one sample to account for the non-strict inequality on both sides
    # in the definition of the last setpoint 
    ref = np.append(ref, ref[-1])

    # compute final time instant
    Tf = np.round(np.sum(n_samples)*Ts, 3)
    
    return (ref, Tf)

def compute_num_steps(ts_sim, Ts, Tf):
    
    # check consistency
    if(round(Ts % ts_sim) != 0):
        raise ValueError("The sample time Ts has to be an integer multiple of the simulation time step ts_sim")
    if(round(Tf % ts_sim) != 0):
        raise ValueError("The simulation time Tf has to be an integer multiple of the simulation time step ts_sim")

    # compute the number of simulation steps
    N_steps    = int(Tf/ts_sim)

    # compute the number of steps for the discrete-time part of the loop
    N_steps_dt = int(Tf/Ts)

    # number of simulation steps every which to update the discrete-time part of the loop 
    n_update   = int(Ts/ts_sim)

    return (N_steps, N_steps_dt, n_update)