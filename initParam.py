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

################################################################################
## CHANGING SECTION
# Manually update the library for the vehicle

m        = 9.4+9.4 # [kg] massa veicolo con mini PC e tutto il resto
b        = l/2 # rear axle distance to CoG
a        = l-b # front axle distance to CoG
J_CoG    = 9.4*a**2+9.4*b**2 # Inertia axis z[kg*m^2]
Fz_Front = 9.4*g # [N]
Fz_Rear  = 9.4*g # [N]
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