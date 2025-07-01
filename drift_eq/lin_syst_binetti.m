function [A,B] = lin_syst_binetti(a,b,track,m,J,Fz,Cy,mu,delta,Fxr,V0,beta0,r0,ratio,fr,K,Fxf)


alpha_f = @(V,beta,r,delta) -atan((V*sin(beta)+a*r)/(V*cos(beta)))+delta;
alpha_r = @(V,beta,r) -atan((V*sin(beta)-b*r)/(V*cos(beta)));
Fymax = @(mu,Fz,Fx) sqrt((mu*Fz)^2-Fx^2);

alpha_sl(1) = atan(3*mu(1)*Fz(1)/Cy(1));
alpha_sl(2) = atan(3*Fymax(mu(2),Fz(2),Fxr)/Cy(2));
        
Fyf = @(V,beta,r,delta,Fx) (Cy(1)*tan(alpha_f(V,beta,r,delta)) ...
            - Cy(1)^2/(3*Fymax(mu(1),Fz(1),Fx))*abs(tan(alpha_f(V,beta,r,delta))).*tan(alpha_f(V,beta,r,delta)) ...
            + Cy(1)^3/(27*Fymax(mu(1),Fz(1),Fx)^2)*(tan(alpha_f(V,beta,r,delta))).^3).*(sign(alpha_sl(1)-abs(alpha_f(V,beta,r,delta)))+1)/2 + Fymax(mu(1),Fz(1),Fx)*sign(alpha_f(V,beta,r,delta)).*(1-sign(alpha_sl(1)-abs(alpha_f(V,beta,r,delta))))/2;
        
Fyr = @(V,beta,r,Fx) (Cy(2)*tan(alpha_r(V,beta,r)) ...
            - Cy(2)^2/(3*Fymax(mu(2),Fz(2),Fx))*abs(tan(alpha_r(V,beta,r))).*tan(alpha_r(V,beta,r)) ...
            + Cy(2)^3/(27*Fymax(mu(2),Fz(2),Fx)^2)*(tan(alpha_r(V,beta,r))).^3).*(sign(alpha_sl(2)-abs(alpha_r(V,beta,r)))+1)/2 + Fymax(mu(2),Fz(2),Fx)*sign(alpha_r(V,beta,r)).*(1-sign(alpha_sl(2)-abs(alpha_r(V,beta,r))))/2;

Mz  = @(ratio,Fx) Fx*track/2*(2*ratio-1);

syms dVdt dbetadt drdt V bet r d F_xr
% dVdt(V,bet,r,d,F_xr) = 1/m*(-Fyf(V,bet,r,d)*sin(d-bet)+(F_xr-0.02*(Fz(1)+Fz(2))-0.2*abs(rad2deg(delta/0.6)))*cos(bet)+Fyr(V,bet,r,F_xr)*sin(bet));
% dbetadt(V,bet,r,d,F_xr) = 1./(m*V).*(Fyf(V,bet,r,d)*cos(d-bet)+Fyr(V,bet,r,F_xr)*cos(bet)-(F_xr-0.02*(Fz(1)+Fz(2))-0.2*abs(rad2deg(delta/0.6)))*sin(bet)) - r;
% drdt(V,bet,r,d,F_xr) = 1/J*(a*Fyf(V,bet,r,d)*cos(d)-b*Fyr(V,bet,r,F_xr)+Mz(ratio,F_xr));

dVdt(V,bet,r,d,F_xr) = 1/m*(Fxf*cos(d-bet)-Fyf(V,bet,r,d,Fxf)*sin(d-bet)+(F_xr-(fr*m*9.81 + K*abs(V*cos(bet)))*sign(V*cos(bet)))*cos(bet)+Fyr(V,bet,r,F_xr)*sin(bet));
dbetadt(V,bet,r,d,F_xr) = 1./(m*V).*(Fxf*sin(d-bet)+Fyf(V,bet,r,d,Fxf)*cos(d-bet)+Fyr(V,bet,r,F_xr)*cos(bet)-(F_xr-(fr*m*9.81 + K*abs(V*cos(bet)))*sign(V*cos(bet)))*sin(bet)) - r;
drdt(V,bet,r,d,F_xr) = 1/J*(a*Fyf(V,bet,r,d,Fxf)*cos(d)-b*Fyr(V,bet,r,F_xr)+Mz(ratio,F_xr)+a*Fxf*sin(d));


A11 = diff(dVdt,V);
A12 = diff(dVdt,bet);
A13 = diff(dVdt,r);

A21 = diff(dbetadt,V);
A22 = diff(dbetadt,bet);
A23 = diff(dbetadt,r);

A31 = diff(drdt,V);
A32 = diff(drdt,bet);
A33 = diff(drdt,r);

B11 = diff(dVdt,d);
B12 = diff(dVdt,F_xr);

B21 = diff(dbetadt,d);
B22 = diff(dbetadt,F_xr);

B31 = diff(drdt,d);
B32 = diff(drdt,F_xr);

A = [A11(V0,beta0,r0,delta,Fxr),A12(V0,beta0,r0,delta,Fxr),A13(V0,beta0,r0,delta,Fxr);A21(V0,beta0,r0,delta,Fxr),A22(V0,beta0,r0,delta,Fxr),A23(V0,beta0,r0,delta,Fxr);A31(V0,beta0,r0,delta,Fxr),A32(V0,beta0,r0,delta,Fxr),A33(V0,beta0,r0,delta,Fxr)];
A = double(A);

B = [B11(V0,beta0,r0,delta,Fxr), B12(V0,beta0,r0,delta,Fxr);B21(V0,beta0,r0,delta,Fxr),B22(V0,beta0,r0,delta,Fxr);B31(V0,beta0,r0,delta,Fxr),B32(V0,beta0,r0,delta,Fxr)];
B = double(B);


