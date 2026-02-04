%% Dimensions

nx=5;  % No. of differential states
nu=2;  % No. of controls
nz=0;  % No. of algebraic states
ny=2+nu; % No. of outputs
nyN=2; % No. of outputs at the terminal point
np=0; % No. of model parameters
nc=0; % No. of general constraints
ncN=0; % No. of general constraints at the terminal point
nbx = 3; % No. of bounds on states
nbu = 2; % No. of bounds on controls

% state and control bounds
nbx_idx = [1 4 5]; % indexs of states which are bounded
nbu_idx = [1 2]; % indexs of controls which are bounded

%% create variables

import casadi.*

states   = SX.sym('states',nx,1);   % differential states
controls = SX.sym('controls',nu,1); % control input
alg      = SX.sym('alg',nz,1);      % algebraic states
params   = SX.sym('paras',np,1);    % parameters
refs     = SX.sym('refs',ny,1);     % references of the first N stages
refN     = SX.sym('refs',nyN,1);    % reference of the last stage
Q        = SX.sym('Q',ny,1);        % weighting matrix of the first N stages
QN       = SX.sym('QN',nyN,1);      % weighting matrix of the last stage
aux      = SX.sym('aux',ny,1);      % auxilary variable
auxN     = SX.sym('auxN',nyN,1);    % auxilary variable

%% Dynamics

m_sprung = 1937.0; % kg 
m_unsprung_f = 48.85; % kg
m_unsprung_r = 51.3; % kg
m = m_sprung + 2*m_unsprung_f + 2*m_unsprung_r;
w = 2.74; % m
a = 1.347; % m
b = w - a;
Izz = 1287.82; % kg/m2
H = 0.328; % m

g = 9.81;% m/s2

% tire-road friction coefficients
muf = 1.22; 
mur = 1.22;

Cyf = 2*6.6e4; % Cornering stiffness [checked with PAC2002]
Cyr = 2*7.9e4; % Cornering stiffness [checked with PAC2002]

Fz_Front_ST = g * m * b / w; % static load at front axle
Fz_Rear_ST = g * m * a / w; % static load at rear axle

steering_ratio = 1/20.8; % steering wheel angle to ground angle


v       = states(1);
beta    = states(2);
r       = states(3);
delta   = states(4);
Fx      = states(5);

d_delta = controls(1);
d_Fx    = controls(2);

delta_ground = delta*steering_ratio;

alpha_f = delta_ground - atan2(v * sin(beta) + a * r, v * cos(beta));
alpha_r = - atan2(v * sin(beta) - b * r, v * cos(beta));


% lateral forces
Fyf = fiala_ca(alpha_f, 0.0, Fz_Front_ST, muf, Cyf); % front lateral force
Fyr = fiala_ca(alpha_r, Fx,  Fz_Rear_ST,  mur, Cyr); % rear lateral force

% equations of motion
dt_v     = (-Fyf*sin(delta_ground - beta) + Fx*cos(beta) + Fyr*sin(beta)) / m;
dt_beta  = ( Fyf*cos(delta_ground - beta) - Fx*sin(beta) + Fyr*cos(beta)) / (m * v) - r;
dt_r     = ( a * Fyf * cos(delta_ground) - b * Fyr ) / Izz;
dt_delta = d_delta;
dt_Fx    = d_Fx;

% explicit ODE RHS
x_dot = [dt_v;
         dt_beta;
         dt_r;
         dt_delta;
         dt_Fx];

% algebraic function
z_fun = [];                   

% implicit ODE: impl_f = 0
xdot = SX.sym('xdot',nx,1);
impl_f = xdot - x_dot;
     
%% Objectives and constraints

% inner objectives
h = [v; ...
     beta; ...
     d_delta; ...
     d_Fx];
hN = h(1:nyN);

% outer objectives
obji = 0.5*(h-refs)'*diag(Q)*(h-refs);
objN = 0.5*(hN-refN)'*diag(QN)*(hN-refN);

obji_GGN = 0.5*(aux-refs)'*(aux-refs);
objN_GGN = 0.5*(auxN-refN)'*(auxN-refN);

% general inequality constraints
general_con = [];
general_con_N = [];

%% NMPC discretizing time length [s]

Ts_st = 1/100.0; % shooting interval time
