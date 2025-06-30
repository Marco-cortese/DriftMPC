initParam;

%% Time of Simulation
time_step_size       = 0.0005; % [s] time step of a simulation

time_simulated       = 30; % [s] time of simulation

%% INITIAL CONDITION

u0                   = 10; % longitudinal velocity [m/s]
v0                   = 2.2; % lateral velocity [m/s]
yaw_rate0            = -deg2rad(10); % yaw rate [rad/s] 

%overwrite yaw
gamma = 0;

% testing
constant_force = 0;
constant_steer = deg2rad(15);
    
%% SIMULATION AND RESULTS
% out = sim("DTM_sim.slx"); % double track model simulation 
out = sim("STM_sim.slx"); % single track model simulation
    

%% SINGLE TRACK MODEL
tout = out.tout; % time vector
pos_CoG = out.pos_CoG; % position of CoG
pos_rear = out.pos_rear; % position of rear wheel
pos_front = out.pos_front; % position of front wheel
steer = out.steer; % steering angle of the car
u = out.u; % longitudinal velocity
v = out.v; % lateral velocity;
yaw_rate = out.yaw_rate; % angular velocity wrt the cog
front_slip_angle = out.front_slip_angle; % front tires (left + right) slip angles
rear_slip_angle = out.rear_slip_angle; % rear tires (left +right) slip angles

% save everything in a mat file
save('DTM_out.mat', 'tout', 'pos_CoG', 'pos_rear', 'pos_front', 'steer', 'u', 'v', 'yaw_rate', 'front_slip_angle', 'rear_slip_angle');

% run a python animation
pyenv(Version="/home/mg/.pyenv/versions/3.12.11/bin/python"); % change this to your python path
pyrunfile('car_anim.py')