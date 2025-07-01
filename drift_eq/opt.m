function [V_opt,delta_opt,Fxr_opt] = opt(a,b,track,m,J,mu,Cy,Fz,beta,R,ratio,fr,K,Fxf)
     
    alpha_f = @(V,beta,r,delta) -atan((V*sin(beta)+a*r)/(V*cos(beta)))+delta;
    alpha_r = @(V,beta,r) -atan((V*sin(beta)-b*r)/(V*cos(beta)));
    Fymax = @(mu,Fz,Fx) sqrt((mu*Fz)^2-Fx^2);
    
    alpha_sl_f = atan(3*mu(1)*Fz(1)/Cy(1));
    alpha_sl_r = @(Fx) atan(3*Fymax(mu(2),Fz(2),Fx)/Cy(2));
    
    Fyf = @(V,beta,r,delta,Fx) (Cy(1)*tan(alpha_f(V,beta,r,delta)) ...
                - Cy(1)^2/(3*Fymax(mu(1),Fz(1),Fx))*abs(tan(alpha_f(V,beta,r,delta))).*tan(alpha_f(V,beta,r,delta)) ...
                + Cy(1)^3/(27*Fymax(mu(1),Fz(1),Fx)^2)*(tan(alpha_f(V,beta,r,delta))).^3).*(sign(alpha_sl_f-abs(alpha_f(V,beta,r,delta)))+1)/2 + Fymax(mu(1),Fz(1),Fx)*sign(alpha_f(V,beta,r,delta)).*(1-sign(alpha_sl_f-abs(alpha_f(V,beta,r,delta))))/2;
            
    Fyr = @(V,beta,r,Fx) (Cy(2)*tan(alpha_r(V,beta,r)) ...
                - Cy(2)^2/(3*Fymax(mu(2),Fz(2),Fx))*abs(tan(alpha_r(V,beta,r))).*tan(alpha_r(V,beta,r)) ...
                + Cy(2)^3/(27*Fymax(mu(2),Fz(2),Fx)^2)*(tan(alpha_r(V,beta,r))).^3).*(sign(alpha_sl_r(Fx)-abs(alpha_r(V,beta,r)))+1)/2 + Fymax(mu(2),Fz(2),Fx)*sign(alpha_r(V,beta,r)).*(1-sign(alpha_sl_r(Fx)-abs(alpha_r(V,beta,r))))/2;
    % ratio to yaw_moment
        
    Mz  = @(ratio,Fx) Fx*track/2*(2*ratio-1);
    function [c,ceq] = nonlinearconst(x)
        c = [];
        % ceq = [1/(m*x(1))*(Fyf(x(1),beta,x(1)/R,x(2))*cos(x(2)-beta)+Fyr(x(1),beta,x(1)/R,x(3))*cos(beta)-(x(3)-0.02*(Fz(1)+Fz(2))-0.2*abs(rad2deg(x(2)/0.6)))*sin(beta)) - x(1)/R;
        %     1/J*(a*Fyf(x(1),beta,x(1)/R,x(2))*cos(x(2))-b*Fyr(x(1),beta,x(1)/R,x(3))+Mz(ratio,x(3))); 
        %     1/m*(-Fyf(x(1),beta,x(1)/R,x(2))*sin(x(2)-beta)+(x(3)-0.02*(Fz(1)+Fz(2))-0.2*abs(rad2deg(x(2)/0.6)))*cos(beta)+Fyr(x(1),beta,x(1)/R,x(3))*sin(beta))];

        ceq = [1/(m*x(1))*(Fxf*sin(x(2)-beta)+Fyf(x(1),beta,x(1)/R,x(2),Fxf)*cos(x(2)-beta)+Fyr(x(1),beta,x(1)/R,x(3))*cos(beta)-(x(3)-(fr*m*9.81 + K*abs(x(1)*cos(beta)))*sign(x(1)*cos(beta)))*sin(beta)) - x(1)/R;
             1/J*(a*Fyf(x(1),beta,x(1)/R,x(2),Fxf)*cos(x(2))-b*Fyr(x(1),beta,x(1)/R,x(3))+Mz(ratio,x(3))+Fxf*a*sin(x(2))); 
             1/m*(Fxf*cos(x(2)-beta)-Fyf(x(1),beta,x(1)/R,x(2),Fxf)*sin(x(2)-beta)+(x(3)-(fr*m*9.81 + K*abs(x(1)*cos(beta)))*sign(x(1)*cos(beta)))*cos(beta)+Fyr(x(1),beta,x(1)/R,x(3))*sin(beta))];
    end
    
    % objective function
    fun_obj = @(x) -x(1);
    % disequality constraint (V>0,Fxr>0)
    A=[-1,0,0;0,0,0;0,0,-1];
    bvec=[0;0;0];
    % nonlinear equality constraint
    nonlcon = @nonlinearconst;
    % initial condition
    %x0=[15,0.05,100];
% 
    % initial condition Canton
    x0=[3,0,1];
    %x0 = [5,0,1];
    Aeq=[];
    beq=[];
    lb=[];
    ub=[];

    x = fmincon(fun_obj,x0,A,bvec,Aeq,beq,lb,ub,nonlcon);
    V_opt = x(1);
    delta_opt = x(2);
    Fxr_opt = x(3);
end
