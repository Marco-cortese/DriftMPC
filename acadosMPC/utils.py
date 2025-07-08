from initParam import * # import constants 
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import casadi as ca

# to visualize videos in the notebook
from IPython.display import Video

mu_f, mu_r = 0.8, 0.8   # [] friction coefficients front and rear
Cyf = 370.36;  # [N/rad] Cornering stiffness front tyre
Cyr = 1134.05; # [N/rad] Cornering stiffness rear tyre


def fiala(alpha, Fx, Fz, mu, Cy):
    # Ensure symbolic compatibility
    Fy_max = ca.sqrt(ca.fmax(1e-6, mu**2 * Fz**2 - Fx**2))
    alphas = ca.atan(3 * Fy_max / Cy)

    tan_alpha = ca.tan(alpha)
    abs_tan_alpha = ca.fabs(tan_alpha)
    sign_alpha = ca.sign(alpha)

    # Compute nonlinear region (|alpha| < alphas)
    Fy_linear = Cy * tan_alpha \
                - (Cy**2 * abs_tan_alpha * tan_alpha) / (3 * Fy_max) \
                + (Cy**3 * tan_alpha**3) / (27 * Fy_max**2)

    # Saturation region (|alpha| ≥ alphas)
    Fy_saturated = Fy_max * sign_alpha

    # Conditional expression using casadi.if_else
    Fy = ca.if_else(ca.fabs(alpha) < alphas, Fy_linear, Fy_saturated)

    return Fy

# tanh approximation
def fiala_tanh(alpha, Fx, Fz, mu, Cy):
    # Avoid invalid sqrt by clamping inside sqrt to >= 0
    Fy_max = ca.sqrt(ca.fmax(1e-6, mu**2 * Fz**2 - Fx**2))
    alphas = ca.atan(Fy_max / Cy)  
    return Fy_max * ca.tanh(alpha / alphas)  

def STM_model()-> AcadosModel:
    
    #set model name
    model_name = "STM_model"
    nx = 3 # differential states: u, v, r
    nu = 2 # inputs: Fx, delta

    # Differential states 
    V       = ca.SX.sym('V', 1, 1)  # total velocity of the car 
    beta    = ca.SX.sym('beta', 1, 1)  # sideslip angle of the car
    r       = ca.SX.sym('r', 1, 1)  # yaw rate of the car

    x = ca.vertcat(V,beta,r)  # state vector

    # Inputs
    delta    = ca.SX.sym('delta', 1, 1)  # steering angle of the car
    Fx       = ca.SX.sym('Fx', 1, 1)  # longitudinal force of the car
   
    u = ca.vertcat(delta, Fx)  # input vector

    # setup symbolic variables for xdot (to be used with IRK integrator)
    V_dot   = ca.SX.sym('vx_dot', 1, 1)
    beta_dot   = ca.SX.sym('vy_dot', 1, 1)
    r_dot    = ca.SX.sym('r_dot', 1, 1)

    xdot = ca.vertcat(V_dot, beta_dot, r_dot)

    # define dynamics
    alpha_f = delta - ca.arctan2(V*ca.sin(beta) + a*r, V*ca.cos(beta)) # slip angle front
    alpha_r = -ca.arctan2(V*ca.sin(beta) - b*r, V*ca.cos(beta)) # slip angle rear
    Fyf = fiala_tanh(alpha_f, 0.0, Fz_Front, mu_f, Cyf) # lateral force front
    Fyr = fiala_tanh(alpha_r, Fx, Fz_Rear, mu_r, Cyr) # lateral force rear
    # Fyf = fiala(alpha_f, 0.0, Fz_Front, mu_f, Cyf) # lateral force front
    # Fyr = fiala(alpha_r, Fx, Fz_Rear, mu_r, Cyr) # lateral force rear


    # Single-track model ODEs in V,beta,r coordinates
    d_V    = (-Fyf * ca.sin(delta - beta) + Fx*ca.cos(beta) + Fyr*ca.sin(beta)) / m
    d_beta = (-Fyf*ca.cos(delta - beta) - Fx*ca.sin(beta) + Fyr*ca.cos(beta)) / (m * V) - r
    d_r    = (a * Fyf * ca.cos(delta) - b * Fyr) / J_CoG

    
    # explicit ODE right hand side (to be used with ERK integrator)
    f_expl = ca.vertcat(d_V, 
                        d_beta, 
                        d_r)

    # implicit dynamics (to be used with IRK integrator)
    f_impl = xdot - f_expl

    # create acados model and fill in all the required fields
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

def clear_previous_simulation():
    variables = globals().copy()
    for v in variables:
        if(type(variables[v]) is AcadosSimSolver or type(variables[v]) is AcadosOcpSolver):
            del globals()[v]

def piecewise_constant(setpoints, setpoints_duration, Ts):

    '''
    Defines the sampled version with sample time Ts of a piecewise constant reference,
    
                 _
                | setpoints(0)      0   <= t < T_1
                | setpoints(1)      T_1 <= t < T_1 + T_2
      ref(t) = <
                | ...
                | setpoints(n-1)    T_1 + ... T_(n-1) <= t <= T_1 + ... + T_n
                 _
    
    where T_1, ..., T_n are the duration of each setpoint, i.e. the entries of setpoints_duration
    
    '''
    # compute the number of samples for each setpoint 
    n_samples = np.rint(setpoints_duration/Ts).astype(int)

    # compute the reference for each setpoint
    ref = setpoints[0]*np.ones((n_samples[0],))

    for i in range(1, len(setpoints)):
        ref = np.append(ref, setpoints[i]*np.ones((n_samples[i],)))

    # add one sample to account for the non-strict inequality on both sides
    # in the definition of the last setpoint 
    ref = np.append(ref, ref[-1])

    # compute final time instant
    Tf = np.round(np.sum(n_samples)*Ts, 3)
    
    return (ref, Tf)

def compute_num_steps(ts_sim, Ts, Tf):
    
    # check consistency
    if(round(Ts % ts_sim) != 0):
        raise ValueError("The sample time Ts has to be an integer multiple of the simulation time step ts_sim")
    if(round(Tf % ts_sim) != 0):
        raise ValueError("The simulation time Tf has to be an integer multiple of the simulation time step ts_sim")

    # compute the number of simulation steps
    N_steps    = int(Tf/ts_sim)

    # compute the number of steps for the discrete-time part of the loop
    N_steps_dt = int(Tf/Ts)

    # number of simulation steps every which to update the discrete-time part of the loop 
    n_update   = int(Ts/ts_sim)

    return (N_steps, N_steps_dt, n_update)