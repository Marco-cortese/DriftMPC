initParam;

%% Time of Simulation
time_step_size       = 0.001; % [s] time step of a simulation

time_simulated       = 40; % [s] time of simulation

%% INITIAL CONDITION

u0                   = 0; % longitudinal velocity [m/s]
v0                   = 0; % lateral velocity [m/s]
yaw_rate0            = 0; %yaw rate [rad/s] 

    
%% SIMULATION AND RESULTS
out                = sim("DTM_sim.slx");
