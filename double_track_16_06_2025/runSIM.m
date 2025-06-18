initParam;

%% Time of Simulation
time_step_size       = 0.001; % [s] time step of a simulation

time_simulated       = 40; % [s] time of simulation

%% INITIAL CONDITION

%_____ Velocity and Yaw Rate
u0                   = 0; 
v0                   = 0;
yaw_rate0            = 0; %[rad/s] yaw rate at start

    
%% SIMULATION AND RESULTS
out                = sim("DTM_sim.slx");
