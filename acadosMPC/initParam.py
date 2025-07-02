# imports
import numpy as np
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

# Define cotangent function
def cot(x): return 1/np.tan(x) # cotangent function

################################################################################
# from Fufy.mat
l = 0.56 # [m] wheelbase
t = 0.41 # [m] car width
g = 9.81 # [m/s^2] gravity acceleration
steering_ratio_f = 0.6 # [-] steering ratio front

################################################################################
## CHANGING SECTION
# Manually update the library for the vehicle

m        = 18.8 # [kg] massa veicolo con mini PC e tutto il resto
b        = l/2 # rear axle distance to CoG
a        = l-b # front axle distance to CoG
J_CoG    = m/2*a**2+m/2*b**2 # Inertia axis z[kg*m^2]
Fz_Front = m*g*b/l # [N]
Fz_Rear  = m*g*a/l # [N]
Fz_Tot   = Fz_Rear + Fz_Front # [N] 
# Second test: height of CoG
H = 160/1000 # [m] 
# h = (Fz_Rear*l/(Fz_Tot)-(l-b))*cot(np.asin(H/l))+(R_r+R_f)/2 # [m] new CoG height
Fz_Front_nominal = Fz_Front # for simulink 
Fz_Rear_nominal = Fz_Rear # for simulink