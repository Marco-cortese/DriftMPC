%% Workspace for Simulink Model
% tommaso.belluco@studenti.unipd.it, 2055261
clear; clc;close all;
load Fufy.mat % file containing all data for car

%% CHANGING SECTION
% Manually update the library for the vehicle

m        = 9.4+9.4; % [kg] veicolo con mini PC e tutto il resto
b        = l/2; % stima del nuovo CoG
a        = l-b;
J_CoG    = 9.4*a^2+9.4*b^2; % [kg*m^2]
Fz_Front = 9.4*g; % [N]
Fz_Rear  = 9.4*g; % [N]
Fz_Tot   = Fz_Rear + Fz_Front; % [N] 
% Second test: height of CoG
Fz_rear  = 9.8*g;
H = 160/1000; % [m] 
h = (Fz_rear*l/(Fz_Tot)-(l-b))*cot(asin(H/l))+(R_r+R_f)/2; % [m] new CoG height
clear H Fz_rear Fz_Rear Fz_Front Fz_Tot

%% CONTROL OF Simulation
time_step_size       = 0.001; % [s] time step of a simulation

time_simulated       = 40; % [s] time of simulation

%% INITIAL CONDITION

%_____ Velocity and Yaw Rate
u0                   = 0; 
v0                   = 0;
yaw_rate0            = 0; %[rad/s] yaw rate at start

%_____ Tires 
R_fl                = R_f; %[m] front left wheel radius
R_fr                = R_f; %[m] front right wheel radius
R_rl                = R_r; %[m] rear left wheel radius
R_rr                = R_r; %[m] rear right wheel radius

%_____ starting position _ 
x0                   = 0; %[m] distance CoG to y-axis (X0)
y0                   = 0; %[m] distance CoG to x-axis (Y0)
gamma                = pi/2; %[rad] starting angle respect to x-axis (gamma0)

%_____ initializing the trajectory
N = 1;
x_plot_CoG           = -200*ones(time_simulated/time_step_size+1,N);
y_plot_CoG           = -200*ones(time_simulated/time_step_size+1,N);

x_plot_rear          = -200*ones(time_simulated/time_step_size+1,N);
y_plot_rear          = -200*ones(time_simulated/time_step_size+1,N);

x_plot_front         = -200*ones(time_simulated/time_step_size+1,N);
y_plot_front         = -200*ones(time_simulated/time_step_size+1,N);

time_plot            = -200*ones(time_simulated/time_step_size+1,N);

V_plot               = -200*ones(time_simulated/time_step_size+1,N);
beta_plot            = -200*ones(time_simulated/time_step_size+1,N);

%___ Kinematic Condition
Toe_fl              = -0*pi/180;
Toe_fr              = -Toe_fl;
Toe_rl              = 0*pi/180;
Toe_rr              = -Toe_rl; 

t_front             = t; %[m] track width front
t_rear              = t; %[m] track width rear
%% SIMULATION AND RESULTS
out                = sim("double_track_09_06_2025.slx");
for t = 1:N
    x_plot_CoG(:,t)    = out.final_position_CoG(:,1);
    y_plot_CoG(:,t)    = out.final_position_CoG(:,2);
    
    x_plot_rear(:,t)   = out.final_position_rear_wheel(:,1);
    y_plot_rear(:,t)   = out.final_position_rear_wheel(:,2);
    
    x_plot_front(:,t)  = out.final_position_front_wheel(:,1);
    y_plot_front(:,t)  = out.final_position_front_wheel(:,2);
    
    time_plot(:,t)     = out.tout;
    
    yaw_rate_plot(:,t) = out.yaw_rate;
    V_plot(:,t)        = out.V;
    beta_plot(:,t)     = out.beta;
    u_plot(:,t)        = out.u;
end
%% PLOT 

%_____ total trajectory of the preparation + drifting condition
figure(2); cla; hold on
set(gca, 'FontSize', 20)
xlabel('sideslip angle (°)', 'FontSize', 20)
ylabel('yaw-rate (°/s)', 'FontSize', 20)
zlabel('Total Velocity (m/s)', 'FontSize', 20)
title('Phase space', 'FontSize', 20)

% Inizializza handle vuoti
h_sim = [];
h_init = [];
h_final = [];
h_drift = [];

for t = 1:N
    % Colore della linea
    color_line = 'k'; % default nero

    % Disegna la linea simulata
    h_line = plot3(rad2deg(beta_plot(:,t)), rad2deg(yaw_rate_plot(:,t)), V_plot(:,t), ...
                   'LineWidth', 0.75, 'LineStyle', '-', 'Color', color_line);
    
    if isempty(h_sim)
        h_sim = h_line;  % Salva solo il primo handle, sarà usato in legenda
    end

    % Punto iniziale
    h2 = plot3(rad2deg(beta_plot(1,t)), rad2deg(yaw_rate_plot(1,t)), V_plot(1,t), ...
               'LineWidth', 1.5, 'LineStyle', 'none', 'Marker', '.', ...
               'Color', 'm', 'MarkerSize', 15);

    % Punto finale
    h3 = plot3(rad2deg(beta_plot(end,t)), rad2deg(yaw_rate_plot(end,t)), V_plot(end,t), ...
               'LineWidth', 1.5, 'LineStyle', 'none', 'Marker', '*', ...
               'Color', 'r', 'MarkerSize', 25);
           
    if isempty(h_init);  h_init = h2;  end
    if isempty(h_final); h_final = h3; end
end

% Punto atteso (drift point)
h4 = plot3(rad2deg(beta_expected), rad2deg(yaw_rate_expected), V_expected, ...
           'LineWidth', 1.5, 'LineStyle', 'none', 'Marker', '+', ...
           'Color', 'g', 'MarkerSize', 50);

% Legenda corretta
legend([h_sim, h_init, h_final, h4], ...
       {'simulated path ', 'initial point', 'final point', 'drift point'}, ...
       'FontSize', 15)

grid on
view(3)

figure(3),cla,hold on
set(gca,fontsize=20)
title('Longitudinal force at rear axle',FontSize=20)
ylabel('Force (N)',FontSize=20)
xlabel('time (s)',FontSize=20)
%plot(out.tout,out.longitudinal_force_estimated,DisplayName='Estimated force',LineStyle='-',LineWidth=1.5,Color='r')
plot(out.tout,out.Fxr,DisplayName='Longitudinal force used',LineStyle='--',Color='k',LineWidth=1.5)
legend (Location="eastoutside")

figure(4),cla,hold on
set(gca,fontsize=20)
title('wheel angle',FontSize=20)
ylabel('delta (°)',FontSize=20)
xlabel('time (s)',FontSize=20)
%plot(out.tout,rad2deg(out.delta_fr),DisplayName='Delta-fr',Marker='*',LineWidth=1.5,Color='b')
plot(out.tout,rad2deg(out.delta_fl),DisplayName='Delta-fl',LineStyle='-',LineWidth=1.5,Color='r')
yline(delta_expected,color='g',DisplayName='Expected',LineWidth=1.5)
legend (Location="eastoutside")

figure(5),cla,hold on
set(gca,fontsize=20)
title('longitudinal speed profile',FontSize=20)
ylabel('delta (°)',FontSize=20)
xlabel('time (s)',FontSize=20)
plot(out.tout,out.u,DisplayName='longitudinal speed',LineStyle='-',LineWidth=1.5,Color='r')
yline(V_expected*cos(beta_expected),color='g',DisplayName='expected',LineWidth=1.5)
legend (Location="eastoutside")

figure(6); cla; hold on
title('Trajectory', FontSize=24)
axis equal
axis([-50 50 -50 +50])
xlabel('x-position [m]', FontSize=24)
ylabel('y-position [m]', FontSize=24)
grid on

% Etichette fittizie per la legenda
h1 = plot(nan, nan, Marker=".", Color='r', MarkerSize=5);
h2 = plot(nan, nan, Marker=".", Color='k', MarkerSize=15);
h3 = plot(nan, nan, Marker=".", Color='b', MarkerSize=5);
h4 = plot(nan, nan, Marker="*", Color='k', MarkerSize=15);
h5 = plot(nan, nan, Marker=".", Color='g', MarkerSize=5);
h6 = plot(nan, nan, Marker="o", Color='k', MarkerSize=15);

legend([h1, h2, h3, h4, h5, h6], ...
       {'Simulated points (CoG)', 'Starting point (CoG)', ...
        'Simulated points (rear)', 'Starting point (rear)', ...
        'Simulated points (front)', 'Starting point (front)'}, ...
       Location="eastoutside", FontSize=20)

% Loop per disegnare i punti simulati e i punti di partenza (10 plot)
for t = 1:N
    % Traccia i punti simulati per CoG, rear e front
    plot(x_plot_CoG(:,t), y_plot_CoG(:,t), Marker=".", Color='r', LineStyle='none', MarkerSize=5, HandleVisibility='off')
    plot(x_plot_CoG(1,t), y_plot_CoG(1,t), Marker=".", Color='k', MarkerSize=15, LineStyle='none', HandleVisibility='off')

    plot(x_plot_rear(:,t), y_plot_rear(:,t), Marker=".", Color='b', LineStyle='none', MarkerSize=5, HandleVisibility='off')
    plot(x_plot_rear(1,t), y_plot_rear(1,t), Marker="*", Color='k', MarkerSize=15, LineStyle='none', HandleVisibility='off')

    plot(x_plot_front(:,t), y_plot_front(:,t), Marker=".", Color='g', LineStyle='none', MarkerSize=5, HandleVisibility='off')
    plot(x_plot_front(1,t), y_plot_front(1,t), Marker="o", Color='k', MarkerSize=15, LineStyle='none', HandleVisibility='off')
end
