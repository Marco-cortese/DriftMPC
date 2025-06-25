%% Workspace for Simulink Model
% tommaso.belluco@studenti.unipd.it, 2055261
clear; clc; close all;
load Fufy.mat % file containing all data for car

%% CHANGING SECTION
% Manually update the library for the vehicle

m        = 9.4+9.4; % [kg] massa veicolo con mini PC e tutto il resto
b        = l/2; % rear axle distance to CoG
a        = l-b; % front axle distance to CoG
J_CoG    = 9.4*a^2+9.4*b^2; % Inertia axis z[kg*m^2]
Fz_Front = 9.4*g; % [N]
Fz_Rear  = 9.4*g; % [N]
Fz_Tot   = Fz_Rear + Fz_Front; % [N] 
% Second test: height of CoG
H = 160/1000; % [m] 
h = (Fz_Rear*l/(Fz_Tot)-(l-b))*cot(asin(H/l))+(R_r+R_f)/2; % [m] new CoG height
clear H Fz_rear Fz_Rear Fz_Front Fz_Tot

%_____ Tires 
R_fl                = R_f; %[m] front left wheel radius
R_fr                = R_f; %[m] front right wheel radius
R_rl                = R_r; %[m] rear left wheel radius
R_rr                = R_r; %[m] rear right wheel radius

%_____ starting position _ 
x0                   = 0; %[m] distance CoG to y-axis (X0)
y0                   = 0; %[m] distance CoG to x-axis (Y0)
gamma                = pi/2; %[rad] starting angle respect to x-axis (gamma0)


%___ Kinematic Condition
Toe_fl              = -0*pi/180;
Toe_fr              = -Toe_fl;
Toe_rl              = 0*pi/180;
Toe_rr              = -Toe_rl; 

t_front             = t; %[m] track width front
t_rear              = t; %[m] track width rear
