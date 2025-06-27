import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from tqdm import tqdm
from scipy.io import loadmat # importing loadmat to read .mat files

# vehicle parameters, from Fufy.mat
fufy = loadmat('Fufy.mat')
a = fufy['a'][0,0] # 'distance in meters from fron to CoG'
b = fufy['b'][0,0] # 'distance in meters from rear to CoG'
l = fufy['l'][0,0] # 'wheelbase in meters'
t = fufy['t'][0,0] # 'track width of the car, equal both sides'

# Load the simulation out .mat file (created by running runSIM.m)
d = loadmat('DTM_out.mat')

tout =              d['tout'].reshape(-1)
pos_CoG =           d['pos_CoG'] 
pos_rear =          d['pos_rear'] 
pos_front =         d['pos_front'] 
u =                 d['u'].reshape(-1) # longitudinal velocity in m/s
v =                 d['v'].reshape(-1) # lateral velocity in m/s
yaw_rate =          d['yaw_rate'].reshape(-1) # yaw rate in rad/s
front_slip_angle =  d['front_slip_angle'] # front tires (left + right) slip angles in rad
rear_slip_angle =   d['rear_slip_angle'] # rear tires (left + right) slip angles in rad
steer =             d['steer'][0,0] # fixed steering for now

front_slip_angle = np.mean(front_slip_angle, axis=1)  # average front slip angle for both tires
rear_slip_angle = np.mean(rear_slip_angle, axis=1)  # average rear slip angle for both tires

# #  plot
# plt.figure(figsize=(10, 10))
# plt.plot(pos_CoG[:, 0], pos_CoG[:, 1], label='CoG', color='blue')
# plt.plot(pos_rear[:, 0], pos_rear[:, 1], label='Rear', color='red')
# plt.plot(pos_front[:, 0], pos_front[:, 1], label='Front', color='green')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.title('Trajectory')
# plt.legend()
# plt.axis('equal')
# # plt.show()

# create an animation 
FPS = 60.0
SPEED = 1.0 # speed multiplier for the animation

# Function to get the car's corner points and wheel shapes in the global frame 
def get_car_shapes(p_cog, p_rear, p_front, a, b, l, t, steer, size_mult=2): # -> car_corners_gf, wrl_gf, wrr_gf, wfl_gf, wfr_gf
    assert p_cog.shape == (2,), "p_cog must be a 2D vector"
    assert p_rear.shape == (2,), "p_rear must be a 2D vector"
    assert p_front.shape == (2,), "p_front must be a 2D vector"

    # plt.figure(figsize=(10, 10))
    # plt.scatter(p_cog[0], p_cog[1], label='p_cog')
    # plt.scatter(p_rear[0], p_rear[1], label='p_rear')
    # plt.scatter(p_front[0], p_front[1], label='p_front')

    yaw = np.arctan2(p_front[1] - p_rear[1], p_front[0] - p_rear[0])  # Calculate yaw angle
    # print(f'Yaw angle: {np.rad2deg(yaw)} degrees')
    r1 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw),  np.cos(yaw)]]) # Rotation matrix for the yaw angle

    # create the car shape in the local frame
    rl = size_mult*np.array([-b, +t/2]) # rear left
    rr = size_mult*np.array([-b, -t/2]) # rear right
    fl = size_mult*np.array([+a, +t/2]) # front left
    fr = size_mult*np.array([+a, -t/2]) # front right
    
    # Create a polygon for the car shape
    car_corners_lf = np.array([rl, rr, fr, fl, rl]).reshape(5, 2)  # Closed shape for the car
    assert car_corners_lf.shape == (5, 2), f"car_corners_lf must be a 5x2 array, not {car_corners_lf.shape}"
    # plt.plot(car_corners_lf[:, 0], car_corners_lf[:, 1], label='car_corners_lf')

    # Transform the car shape to the global frame
    car_corners_gf = car_corners_lf @ r1.T + p_cog  # rototranslation to global frame
    # plt.plot(car_corners_gf[:, 0], car_corners_gf[:, 1], label='car_corners_gf')

    # draw the 4 wheels as 4 rectangles
    wheel_w, wheel_h = 0.1, 0.2  # wheel width and height
    wheel = size_mult* np.array([  
                        [-wheel_h/2, -wheel_w/2],
                        [+wheel_h/2, -wheel_w/2],
                        [+wheel_h/2, +wheel_w/2],
                        [-wheel_h/2, +wheel_w/2],
                        [-wheel_h/2, -wheel_w/2]])  # Closed shape for the wheel
    assert wheel.shape == (5, 2), f"wheel must be a 5x2 array, not {wheel.shape}"
    # plt.plot(wheel[:, 0], wheel[:, 1], label='wheel')
    
    r2 = np.array([[np.cos(steer), -np.sin(steer)], [np.sin(steer),  np.cos(steer)]]) # Rotation matrix for the steering angle
    assert rl.shape == (2,) and wheel.shape == (5, 2), f"rl must be a 2D vector and wheel must be a 5x2 array, not {rl.shape} and {wheel.shape}"
    wrl_lf = rl + wheel  # rear left wheel
    wrr_lf = rr + wheel  # rear right wheel
    wfl_lf = fl + wheel @ r2.T  # front left wheel
    wfr_lf = fr + wheel @ r2.T  # front right wheel
    assert wrl_lf.shape == (5, 2), f"wrl_lf must be a 5x2 array, not {wrl_lf.shape}"
    assert wrr_lf.shape == (5, 2), f"wrr_lf must be a 5x2 array, not {wrr_lf.shape}"
    assert wfl_lf.shape == (5, 2), f"wfl_lf must be a 5x2 array, not {wfl_lf.shape}"
    assert wfr_lf.shape == (5, 2), f"wfr_lf must be a 5x2 array, not {wfr_lf.shape}"

    # plt.plot(wrl_lf[:, 0], wrl_lf[:, 1], label='wrl_lf')
    # plt.plot(wrr_lf[:, 0], wrr_lf[:, 1], label='wrr_lf')
    # plt.plot(wfl_lf[:, 0], wfl_lf[:, 1], label='wfl_lf')
    # plt.plot(wfr_lf[:, 0], wfr_lf[:, 1], label='wfr_lf')
    
    # rototranslation to global frame
    wrl_gf = wrl_lf @ r1.T + p_cog
    wrr_gf = wrr_lf @ r1.T + p_cog  
    wfl_gf = wfl_lf @ r1.T + p_cog
    wfr_gf = wfr_lf @ r1.T + p_cog

    # plt.plot(wrl_gf[:, 0], wrl_gf[:, 1], label='wrl_gf')
    # plt.plot(wrr_gf[:, 0], wrr_gf[:, 1], label='wrr_gf')
    # plt.plot(wfl_gf[:, 0], wfl_gf[:, 1], label='wfl_gf')
    # plt.plot(wfr_gf[:, 0], wfr_gf[:, 1], label='wfr_gf')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.title('Car Position')
    # plt.legend()
    # plt.axis('equal')
    # plt.show()  # Show the plot with the car shape and wheels

    return car_corners_gf, wrl_gf, wrr_gf, wfl_gf, wfr_gf

# create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(np.min(pos_CoG[:, 0]) - 2, np.max(pos_CoG[:, 0]) + 2)
ax.set_ylim(np.min(pos_CoG[:, 1]) - 2, np.max(pos_CoG[:, 1]) + 2)
ax.set_aspect('equal')
# ax.plot(pos_CoG[:, 0], pos_CoG[:, 1])
# ax.plot(pos_rear[:, 0], pos_rear[:, 1])
# ax.plot(pos_front[:, 0], pos_front[:, 1])


# slip = ax.scatter(pos_CoG[:, 0], pos_CoG[:, 1], c=np.abs(np.rad2deg(front_slip_angle)), s=2, cmap='viridis')
# cbar = plt.colorbar(slip, ax=ax, label='Front slip angle α [deg]')

slip = ax.scatter(pos_CoG[:, 0], pos_CoG[:, 1], c=np.abs(np.rad2deg(rear_slip_angle)), s=2, cmap='viridis')
cbar = plt.colorbar(slip, ax=ax, label='Rear slip angle α [deg]', orientation='horizontal')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Car Animation')
# plt.tight_layout()

# create initial plots for the car and wheels
car, wrl, wrr, wfl, wfr = get_car_shapes(pos_CoG[0], pos_rear[0], pos_front[0], a, b, l, t, steer)

car_plot, = ax.plot(car[:, 0], car[:, 1], color='yellow')
wrl_plot, = ax.plot(wrl[:, 0], wrl[:, 1], color='orange')
wrr_plot, = ax.plot(wrr[:, 0], wrr[:, 1], color='orange')
wfl_plot, = ax.plot(wfl[:, 0], wfl[:, 1], color='orange')
wfr_plot, = ax.plot(wfr[:, 0], wfr[:, 1], color='orange')

anim_cog = pos_CoG[::int(FPS/SPEED), :]
anim_rear = pos_rear[::int(FPS/SPEED), :]
anim_front = pos_front[::int(FPS/SPEED), :]
# anim_steer = steer[::int(FPS/SPEED)]

def update(frame):
    # Update the car and wheels positions    
    cog, rear, front = anim_cog[frame], anim_rear[frame], anim_front[frame]
    assert cog.shape == (2,), f"cog must be a 2D vector, not {cog.shape}"
    assert rear.shape == (2,), f"rear must be a 2D vector, not {rear.shape}"
    assert front.shape == (2,), f"front must be a 2D vector, not {front.shape}"
    car, wrl, wrr, wfl, wfr = get_car_shapes(cog, rear, front, a, b, l, t, steer)
    
    # Update the plots
    car_plot.set_data(car[:, 0], car[:, 1])
    wrl_plot.set_data(wrl[:, 0], wrl[:, 1])
    wrr_plot.set_data(wrr[:, 0], wrr[:, 1])
    wfl_plot.set_data(wfl[:, 0], wfl[:, 1])
    wfr_plot.set_data(wfr[:, 0], wfr[:, 1])
    
    # print(f'anim: [{frame+1}/{len(anim_cog)}] [{100*(frame+1)/len(anim_cog):.1f}%]         ', end='\r')

    return car_plot, wrl_plot, wrr_plot, wfl_plot, wfr_plot

# Create the animation
anim = FuncAnimation(fig, update, frames=len(anim_cog), interval=1000/FPS, blit=True)

# anim.save('car_animation.gif', fps=FPS, dpi=50)  # save animation as gif
# anim.save('car_animation.mp4', fps=FPS, extra_args=['-vcodec', 'libx264']) # save animation as mp4

plt.show() # show the animation
