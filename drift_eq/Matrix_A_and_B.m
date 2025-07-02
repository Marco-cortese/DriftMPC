clc,clear all, close all
%% PARAMETERS
m           = 18.8;          % [kg] vehicle mass
a           = 0.28;        % [m] 
b           = 0.28;        % [m]
l           = a+b;           % [m]
J           = 9.4*a^2+9.4*b^2;% [kgm^2] yaw moment of inertia
C_alpha_f   = 370;           % [N/rad] front axle cornering stiffness
C_alpha_r   = 1134;          % [N/rad] rear axle cornering stiffness 
g           = 9.81;
Cy          = [C_alpha_f,C_alpha_r];
muf         = 0.8;           % front friction coefficient
mur         = 0.8;          % rear friction coefficient
mu          = [muf,mur];
Fzf         = m*g*b/l;       % front tyre static load (article)
Fzr         = m*g*a/l;       % rear tyre static load (article)
Fz          = [Fzf,Fzr];
m           = 21.4;
track       = 0.41;          % track width

%_____ Resistance
fr      = 0.049;
K_fr    = 2.85;
Fxf     = 0;
ratio   = 65/100;
% %% FORMULA
% alpha_f = @(V,beta,r,delta) -atan((V.*sin(beta)+a*r)./(V.*cos(beta)))+delta;
% alpha_r = @(V,beta,r) -atan((V.*sin(beta)-b*r)./(V.*cos(beta)));
% Fymax = @(mu,Fz,Fx) sqrt((mu*Fz)^2-Fx.^2);
% 
% alpha_sl_f = atan(3*mu(1)*Fz(1)/Cy(1));
% alpha_sl_r = @(Fx) atan(3*Fymax(mu(2),Fz(2),Fx)/Cy(2));
% 
% % Fy = Fy(V,beta,r)        
% Fyf = @(V,beta,r,delta) (Cy(1)*tan(alpha_f(V,beta,r,delta)) ...
%             - Cy(1)^2/(3*mu(1)*Fz(1))*abs(tan(alpha_f(V,beta,r,delta))).*tan(alpha_f(V,beta,r,delta)) ...
%             + Cy(1)^3/(27*mu(1)^2*Fz(1)^2)*(tan(alpha_f(V,beta,r,delta))).^3).*(sign(alpha_sl_f-abs(alpha_f(V,beta,r,delta)))+1)/2 + mu(1)*Fz(1)*sign(alpha_f(V,beta,r,delta)).*(1-sign(alpha_sl_f-abs(alpha_f(V,beta,r,delta))))/2;
% 
% Fyr = @(V,beta,r,Fx) (Cy(2)*tan(alpha_r(V,beta,r)) ...
%             - Cy(2)^2./(3*Fymax(mu(2),Fz(2),Fx)).*abs(tan(alpha_r(V,beta,r))).*tan(alpha_r(V,beta,r)) ...
%             + Cy(2)^3./(27*Fymax(mu(2),Fz(2),Fx).^2).*(tan(alpha_r(V,beta,r))).^3).*(sign(alpha_sl_r(Fx)-abs(alpha_r(V,beta,r)))+1)/2 + Fymax(mu(2),Fz(2),Fx).*sign(alpha_r(V,beta,r)).*(1-sign(alpha_sl_r(Fx)-abs(alpha_r(V,beta,r))))/2;
% 
% % Fy = Fy(alpha)
% Fyf_alpha = @(alpha) (Cy(1)*tan(alpha) ...
%             - Cy(1)^2/(3*mu(1)*Fz(1))*abs(tan(alpha)).*tan(alpha) ...
%             + Cy(1)^3/(27*mu(1)^2*Fz(1)^2)*(tan(alpha)).^3).*(sign(alpha_sl_f-abs(alpha))+1)/2 + mu(1)*Fz(1)*sign(alpha).*(1-sign(alpha_sl_f-abs(alpha)))/2;
% 
% Fyr_alpha = @(alpha,Fx) (Cy(2)*tan(alpha) ...
%             - Cy(2)^2./(3*Fymax(mu(2),Fz(2),Fx)).*abs(tan(alpha)).*tan(alpha) ...
%             + Cy(2)^3./(27*Fymax(mu(2),Fz(2),Fx).^2).*(tan(alpha)).^3).*(sign(alpha_sl_r(Fx)-abs(alpha))+1)/2 + Fymax(mu(2),Fz(2),Fx).*sign(alpha).*(1-sign(alpha_sl_r(Fx)-abs(alpha)))/2;
%% EQUILIBRIA POINTS
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
yellow = [0.9290 0.6940 0.1250];
purple = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];
light_blue = [0.3010 0.7450 0.9330];
red = [0.6350 0.0780 0.1840];

colors = [blue;orange;yellow;purple;green;light_blue;red];

R=3.5; % Select cornering radius

figure('units','centimeters','position',[1,1,12,14]);
subplot(3,1,1)
hold on
grid on
box on
xlabel('$\beta (deg)$','FontSize',14)
ylabel('V (m/s)','FontSize',14)
title(['R = ' num2str(R) ' m'], 'FontSize',14)
% legend('FontSize',12,'Location','southeast') % Add entries manually
subplot(3,1,2)
hold on
grid on
box on
xlabel('$\beta$ (deg)','FontSize',14)
ylabel('$\delta$ (deg)','FontSize',14)
subplot(3,1,3)
hold on
grid on
box on
xlabel('$\beta$ (deg)','FontSize',14)
ylabel('$F_{xr}$ (N)','FontSize',14)

beta_span = [deg2rad(-35:1:-2),deg2rad(-1.8:0.2:0)];

V_opt = [];
delta_opt = [];
Fxr_opt = [];

for i=1:length(beta_span)
    %[V_opt(i),delta_opt(i),Fxr_opt(i)] = opt(a,b,m,J,mu,Cy,Fz,beta_span(i),R);
    mu = [muf, mur];
    Fz = [Fzf, Fzr];
    Cy = [C_alpha_f, C_alpha_r];
    [V_opt(i), delta_opt(i), Fxr_opt(i)] = opt(a,b,track,m,J,mu,Cy,Fz,beta_span(i),R,ratio,fr,K_fr,Fxf);
 
   % Ai = lin_syst_binetti(a,b,m,J,Fz,Cy,mu,delta_opt(i),Fxr_opt(i),V_opt(i),beta_span(i),V_opt(i)/R);
    Ai = lin_syst_binetti(a, b, track, m, J, Fz, Cy, mu, delta_opt(i), Fxr_opt(i), V_opt(i), beta_span(i), V_opt(i)/R, ratio,fr,K_fr,Fxf);
    % 
    % disp(['mu = ', mat2str(mu), ' - size: ', num2str(size(mu))]);
    % disp(['Fz = ', mat2str(Fz), ' - size: ', num2str(size(Fz))]);
    % disp(['Cy = ', mat2str(Cy), ' - size: ', num2str(size(Cy))]);
    [Vi,Di]=eig(Ai);
    Di11(i)=Di(1,1);
    Di22(i)=Di(2,2);
    Di33(i)=Di(3,3);
    %alphar(i) = alpha_r(V_opt(i),beta_span(i),V_opt(i)/R);
    if Di(1,1)<0 && Di(2,2)<0
        subplot(3,1,1)
        plot(rad2deg(beta_span(i)),V_opt(i),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%             plot(rad2deg(alphar(i)),V_opt(i),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%              %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
        subplot(3,1,2)
        plot(rad2deg(beta_span(i)),rad2deg(delta_opt(i)),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%             plot(rad2deg(alphar(i)),rad2deg(delta_opt(i)),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%              %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
        subplot(3,1,3)
        plot(rad2deg(beta_span(i)),Fxr_opt(i),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%             plot(rad2deg(alphar(i)),Fxr_opt(i),'o','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',green,'HandleVisibility','off')
%              %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
    else
        if sign(delta_opt(i))*sign(V_opt(i)/R)>0
            subplot(3,1,1)
            plot(rad2deg(beta_span(i)),V_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','on')
%                 plot(rad2deg(alphar(i)),V_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
            subplot(3,1,2)
            plot(rad2deg(beta_span(i)),rad2deg(delta_opt(i)),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','on')
%                 plot(rad2deg(alphar(i)),rad2deg(delta_opt(i)),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
            subplot(3,1,3)
            plot(rad2deg(beta_span(i)),Fxr_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','on')
%                 plot(rad2deg(alphar(i)),Fxr_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',yellow,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
        else
            subplot(3,1,1)
            plot(rad2deg(beta_span(i)),V_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','on')
%                 plot(rad2deg(alphar(i)),V_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
            subplot(3,1,2)
            plot(rad2deg(beta_span(i)),rad2deg(delta_opt(i)),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','on')
%                 plot(rad2deg(alphar(i)),rad2deg(delta_opt(i)),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
            subplot(3,1,3)
            plot(rad2deg(beta_span(i)),Fxr_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','off')
%                 plot(rad2deg(alphar(i)),Fxr_opt(i),'d','MarkerSize',7,'MarkerEdgeColor','none','MarkerFaceColor',red,'HandleVisibility','off')
%                  %% If you want rear tire slip angle on the x-axis, uncomment this line and comment the above
        end
    end
end



%% MATRIX

beta_eq = [deg2rad(-25)]; % sideslip angles to be plotted
R       = 3.5; % Cornering radius

[V_eq,delta_eq,Fxr_eq] = opt(a,b,track,m,J,mu,Cy,Fz,beta_eq,R,ratio,fr,K_fr,Fxf); % Function that finds driver inputs, given beta and R
r_eq=V_eq/R;

u_eq = V_eq * cos(beta_eq);
v_eq = V_eq * sin(beta_eq);

fprintf("Equilibrium --> u: %.2f, v: %.2f, r: %.2f, delta: %.2f, Fxr: %.2f \n", u_eq, v_eq, r_eq, delta_eq, Fxr_eq);
fprintf("Equilibrium --> V: %.2f, beta: %.2f, r: %.2f, delta: %.2f, Fxr: %.2f \n", V_eq, beta_eq, r_eq, delta_eq, Fxr_eq);

[A,B] = lin_syst_binetti(a,b,track,m,J,Fz,Cy,mu,delta_eq,Fxr_eq,V_eq,beta_eq,r_eq,ratio,fr,K_fr,Fxf);
%[A,B] = lin_syst_binetti_2(a,b,track,m,J,Fz,Cy,mu,delta_eq,Fxr_eq,V_eq*cos(beta_eq),V_eq*sin(beta_eq),r_eq,ratio,fr,K_fr,Fxf);

%___
[V,D]=eig(A);

if D(1,1)<0 && D(2,2)<0 && D(3,3)<0
        disp("this point is a normal turn")
else
        if sign(delta_eq)*sign(r_eq)>0
            disp("this is unstable but not a drift point")
        else
            disp("this is a drift point")
        end
end
%___ LQR Design

% According to Bryson's rule, the Q, R weights are defined in a diagonal matrix with the terms as the
% reciprocal of the square of the maximum deviation from the equilibrium in
% terms of states (for Q) and input (R).

Q11 = 1/(1)^2;
Q22 = 1/(0.5)^2; 
Q33 = 1/(0.5)^2;

R11 = 1/(0.5)^2; 
R22 = 1/(10)^2;

Q = diag([Q11,Q22,Q33]);
R = diag([R11,R22]);

% Calcolo del guadagno LQR
K = lqr(A, B, Q, R);

% Closed loop
sys_cl = ss(A-B*K, zeros(3,0), eye(3), zeros(3,0));
x0 = [2 -1 0]';
initial(sys_cl, x0)


% Salvo K per usarlo in Simulink
Fxf_eq = Fxf;
save(['gainLQR_drift.mat'],'A','B', 'K','V_eq','delta_eq','Fxr_eq','r_eq','beta_eq','ratio','track','fr','K_fr','Fxf_eq');





