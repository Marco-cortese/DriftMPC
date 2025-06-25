initParam;

%% Time of Simulation
time_step_size       = 0.001; % [s] time step of a simulation

time_simulated       = 40; % [s] time of simulation

%% INITIAL CONDITION

u0                   = 0; % longitudinal velocity [m/s]
v0                   = 0; % lateral velocity [m/s]
yaw_rate0            = 0; % yaw rate [rad/s] 

    
%% SIMULATION AND RESULTS
out                = sim("DTM_sim.slx");


%% PLOT RESULTS
tout = out.tout; % time vector
pos_CoG = out.pos_CoG; % position of CoG
pos_rear = out.pos_rear; % position of rear wheel
pos_front = out.pos_front; % position of front wheel
steer = out.steer; % steering angle of the car
u = out.u; % longitudinal velocity
v = out.v; % lateral velocity;

% save everything in a mat file
save('DTM_out.mat', 'tout', 'pos_CoG', 'pos_rear', 'pos_front', 'steer', 'u', 'v');

figure(1);
plot(pos_CoG(:,1), pos_CoG(:,2)); hold on;
plot(pos_rear(:,1), pos_rear(:,2));
plot(pos_front(:,1), pos_front(:,2));
xlabel('X Position [m]'); ylabel('Y Position [m]'); title('Trajectory of the Vehicle');
axis equal; grid on; legend('CoG', 'Rear Wheel', 'Front Wheel');



