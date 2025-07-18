from mc_utils import *
from scipy.linalg import block_diag

# casadi/acados model
import casadi as ca
from casadi import SX, sqrt, atan, tan, sin, cos, tanh, atan2, fabs, vertcat, if_else, sign
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
# suppress warnings acados
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='acados_template')





# tire model functions
def fiala_tanh_ca(alpha, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    alpha_s = atan(Fy_max/Cy) # maximum slip angle
    return Fy_max * tanh(alpha / alpha_s) # tanh approximation
def fiala_ca(alpha, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    Fy_lin = Cy * tan(alpha) - Cy**2 * fabs(tan(alpha)) * tan(alpha) / (3 * Fy_max) + Cy**3 * tan(alpha)**3 / (27 * Fy_max**2)
    Fy_sat = Fy_max * sign(alpha)
    return if_else(fabs(alpha) < atan(3*Fy_max/Cy), Fy_lin, Fy_sat)
def pacejka_ca(α, Fx, Fz, μ, Cy):
    Fy_max = sqrt(μ**2 * Fz**2 - Fx**2) # maximum lateral force
    pB, pC, pD, pE = 4,1.8,Fy_max,-1 # pacejka_ca parameters
    return pD * sin(pC * atan(pB * α - pE * (pB*α - atan(pB*α))))




def STM_model_dt_inputs(tire=fiala_tanh_ca, μ_err=0.0, stiff_k=1.0, J_k=1.0):
    # variables
    v = SX.sym('v') # velocity
    beta = SX.sym('beta') # sideslip angle
    r = SX.sym('r') # yaw rate
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, beta, r, delta, Fx) # state vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire slip
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front, μf+μ_err, Cyf*stiff_k) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear, μr+μ_err, Cyr*stiff_k) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / (J_CoG*J_k) # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def DTM_model_dt_inputs_sim(tire=fiala_tanh_ca, μ_err=0.05, stiff_k=1.05, J_k=1.0):
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # derivative of wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # derivative of rear longitudinal force


    x = vertcat(v, beta, r, delta, Fx) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_fl = delta - atan2(v*sin(beta) + a*r, v*cos(beta) - r*t/2)
    alpha_fr = delta - atan2(v*sin(beta) + a*r, v*cos(beta) + r*t/2)
    alpha_rl = - atan2(v*sin(beta) - a*r, v*cos(beta) - r*t/2)
    alpha_rr = - atan2(v*sin(beta) - a*r, v*cos(beta) + r*t/2)

    # lateral forces
    nu = 0.5
    Fxl = nu*Fx
    Fxr = (1-nu)*Fx
    Fy_fl = tire(alpha_fl, 0.0, Fz_Front/2, μf+μ_err, (Cyf/2)*stiff_k) # lateral force front-left
    Fy_fr = tire(alpha_fr, 0.0, Fz_Front/2, μf+μ_err, (Cyf/2)*stiff_k) # lateral force front-right
    Fy_rl = tire(alpha_rl, Fxl, Fz_Rear/2, μr+μ_err, (Cyr/2)*stiff_k) # lateral force rear
    Fy_rr = tire(alpha_rr, Fxr, Fz_Rear/2, μr+μ_err, (Cyr/2)*stiff_k) # lateral force rear


    # define the symbolic equations of motion
    dt_v = (-(Fy_fl + Fy_fr)*sin(delta-beta) + Fx*cos(beta) + (Fy_rl + Fy_rr)*sin(beta)) / m # V dot
    dt_beta = (+(Fy_fl + Fy_fr)*cos(delta-beta) - Fx*sin(beta) + (Fy_rl + Fy_rr)*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*(Fy_fl + Fy_fr)*cos(delta) - b*(Fy_rl + Fy_rr) + (Fxr - Fxl)*t/2) / (J_CoG*J_k) # r dot

    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear-left longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_dt_inputs_sim(tire=fiala_tanh_ca, μ_err=0.05, stiff_k=1.05, J_k=1.0):
    # variables
    v = SX.sym('v') # velocity
    # v̇ = SX.sym('v̇') # velocity dot
    beta = SX.sym('beta') # sideslip angle
    # β̇ = SX.sym('β̇') # sideslip angle dot
    r = SX.sym('r') # yaw rate
    # ṙ = SX.sym('ṙ') # yaw rate dot
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx') # rear longitudinal force

    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx') # change in rear longitudinal force

    x = vertcat(v, beta, r, delta, Fx) # state vector
    # ẋ = vertcat(v̇, β̇, ṙ) # state dot vector
    u = vertcat(d_delta, d_Fx) # u input vector 

    # tire model
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

    # lateral forces
    Fyf = tire(alpha_f, 0.0, Fz_Front, μf+μ_err, Cyf*stiff_k) # lateral force front
    Fyr = tire(alpha_r, Fx, Fz_Rear, μr+μ_err, Cyr*stiff_k) # lateral force rear
    Fxr = Fx # rear longitudinal force

    # define the symbolic equations of motion
    dt_v = (-Fyf*sin(delta-beta) + Fxr*cos(beta) + Fyr*sin(beta)) / m # V dot
    dt_beta = (+Fyf*cos(delta-beta) - Fxr*sin(beta) + Fyr*cos(beta)) / (m*v) - r # β dot
    dt_r = (a*Fyf*cos(delta) - b*Fyr) / (J_CoG*J_k) # r dot
    dt_delta = d_delta # change in wheel angle (on the road)
    dt_Fx = d_Fx # change in rear longitudinal force

    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx) # equations of motion

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    # model.f_impl_expr = ẋ - dx  
    model.f_expl_expr = dx 
    model.x = x  # state vector
    # model.xdot = ẋ  # state dot vector
    model.u = u  # u input vector
    return model

def STM_model_LT_sim(tire=fiala_tanh_ca, μ_err=0.05, stiff_k=1.05, J_k=1.0):
    # State variables (x)
    v = SX.sym('v')       # velocity magnitude
    beta = SX.sym('beta') # sideslip angle
    r = SX.sym('r')       # yaw rate
    delta = SX.sym('delta') # wheel angle (on the road)
    Fx = SX.sym('Fx')     # rear longitudinal force (treated as a state)
    x = vertcat(v, beta, r, delta, Fx)

    # State derivatives (x_dot)
    v_dot = SX.sym('v_dot')
    beta_dot = SX.sym('beta_dot')
    r_dot = SX.sym('r_dot')
    delta_dot = SX.sym('delta_dot')
    Fx_dot = SX.sym('Fx_dot')
    x_dot = vertcat(v_dot, beta_dot, r_dot, delta_dot, Fx_dot)

    # Input variables (u)
    d_delta = SX.sym('d_delta') # change in wheel angle (on the road)
    d_Fx = SX.sym('d_Fx')       # change in rear longitudinal force
    u = vertcat(d_delta, d_Fx)

    # Algebraic variables (z)
    ax = SX.sym('ax')     # longitudinal acceleration (body frame)
    ay = SX.sym('ay')     # lateral acceleration (body frame)
    Fzf = SX.sym('Fzf')
    Fzr = SX.sym('Fzr')
    z = vertcat(ax, ay, Fzf, Fzr)

    # --- Intermediate Calculations ---
    
    # Slip angles
    alpha_f = delta - atan2(v*sin(beta) + a*r, v*cos(beta))
    alpha_r = -atan2(v*sin(beta) - b*r, v*cos(beta))

    # Tire forces (using algebraic Fzf, Fzr)
    Fyf = tire(alpha_f, 0.0, Fzf, μf+μ_err, Cyf*stiff_k)
    Fyr = tire(alpha_r, Fx, Fzr, μr+μ_err, Cyr*stiff_k)
    Fxr = Fx # Rear longitudinal force is a state

    # --- System Equations (Implicit Form: f_impl = 0) ---
    
    # 1. Algebraic Equations (alg_eqs = 0)
    alg_eqs = []
    
    # DYNAMICS: Define how forces create body-frame accelerations (ax, ay)
    # This is Newton's Second Law.
    alg_eqs.append(m * ax - (Fxr - Fyf*sin(delta)))
    alg_eqs.append(m * ay - (Fyf*cos(delta) + Fyr))

    # LOAD TRANSFER: Define vertical forces based on longitudinal acceleration (ax)
    Fz_LT = m * ax * h / l
    alg_eqs.append(Fzf - (Fz_Front - Fz_LT))
    alg_eqs.append(Fzr - (Fz_Rear + Fz_LT))
    
    # 2. Differential Equations (x_dot - dx = 0)
    
    # KINEMATICS: Define how state derivatives relate to body-frame accelerations
    dt_v = ax * cos(beta) + ay * sin(beta)
    # Add a safety term for low speeds to prevent division by zero
    dt_beta = (ay * cos(beta) - ax * sin(beta)) / ca.fmax(v, 0.1) - r
    
    # Remaining dynamics
    dt_r = (a * Fyf * cos(delta) - b * Fyr) / (J_CoG*J_k)
    dt_delta = d_delta
    dt_Fx = d_Fx
    
    dx = vertcat(dt_v, dt_beta, dt_r, dt_delta, dt_Fx)
    
    # Combine into the final implicit model expression
    f_impl = vertcat(x_dot - dx, *alg_eqs)

    # create the model
    model = AcadosModel()
    model.name='stm_model'
    model.f_impl_expr = f_impl
    # model.f_expl_expr = dx 
    model.x = x  # state vector
    model.z = z # algebraic vector
    model.xdot = x_dot  # state dot vector
    model.u = u  # u input vector
    return model

# utils
def clear_previous_simulation():
    vars = globals().copy()
    for v in globals().copy():
        if(type(vars[v]) is AcadosSimSolver or type(vars[v]) is AcadosOcpSolver):
            del globals()[v]
    # clear the c_generated_code directory
    if os.path.exists('c_generated_code'):
        for f in os.listdir('c_generated_code'):
            file_path = os.path.join('c_generated_code', f)
            if os.path.isfile(file_path): os.remove(file_path)
    # # clear __pycache__ directories
    # for root, dirs, files in os.walk('.'):
    #     for d in dirs:
    #         if d == '__pycache__':
    #             dir_path = os.path.join(root, d)
    #             for f in os.listdir(dir_path):
    #                 file_path = os.path.join(dir_path, f)
    #                 if os.path.isfile(file_path): os.remove(file_path)
    #             os.rmdir(dir_path)
    # # remove acados_ocp.json and acados_sim.json files
    # if os.path.exists('acados_ocp.json'):
    #     os.remove('acados_ocp.json')    
    # if os.path.exists('acados_sim.json'):
    #     os.remove('acados_sim.json')
    return


def piecewise_constant(vals, durs, dt):
    if type(vals) is list: vals = np.array(vals).reshape(-1, 1)  # ensure vals is a column vector
    assert vals.shape[0] == len(durs), "vals and durs must have the same length"
    assert vals.ndim == 2, "vals must be a 2D array"
    tot_samples = int(round(np.sum(durs) / dt))
    durs_n = [int(round(d/dt)) for d in durs]
    pc = np.concatenate([np.full((n,vals.shape[1]), v) for v, n in zip(vals, durs_n)], axis=0)
    return np.append(pc, vals[-1:], axis=0), tot_samples * dt

def compute_num_steps(ts_sim, Ts, Tf):
    # check consistency
    assert abs(Ts/ts_sim - round(Ts/ts_sim)) < 1e-6, f"ts_sim {ts_sim} must be a divisor of Ts {Ts}"
    assert abs(Tf/ts_sim - round(Tf/ts_sim)) < 1e-6, f"ts_sim {ts_sim} must be a divisor of Tf {Tf}"
    N_steps    = int(round(Tf/ts_sim)) # compute the number of simulation steps
    N_steps_dt = int(round(Tf/Ts)) # compute the number of steps for the discrete-time part of the loop
    n_update   = int(round(Ts/ts_sim)) # number of simulation steps every which to update the discrete-time part of the loop 
    return (N_steps, N_steps_dt, n_update)


class Test():
    def __init__(self, 
                    title='Default Test',
                    Ts=0.01, 
                    N=100,
                    T_tot=9.0,
                    model_f=STM_model_dt_inputs, 
                    tire_f=fiala_tanh_ca, 
                    sim_model_f=STM_model_dt_inputs, 
                    sim_tire_f=fiala_tanh_ca, 
                    μ_err=0.0, 
                    stiff_k=1.0, 
                    J_k=1.0,
                    x_eq=[4.486209860862883, -0.4363323129985824, 1.4954032869542941, -0.11596738898598893, 46.64426852037662],
                    u_eq=[0.0, 0.0],
                    delta_lb=-MAX_DELTA,
                    delta_ub=MAX_DELTA,
                    Fx_lb=MIN_FX,
                    Fx_ub=MAX_FX,
                    x0=[2, 0, 0, 0, 0],
                    u0=[0, 0],
                    mpc_ws=[1e2, 1e3, 0, 0, 0, 1e1, 1e-2],
                    V_ref=([0], [1]),
                    beta_ref=([0], [1]),
                    r_ref=([10], [1]),
                    delta_ref=([0], [1]),
                    Fx_ref=([0], [1]),
                    shifting=False,
                    ref_preview=True,
                    V_noise=(0,0), # (0, 0.3), #
                    beta_noise=(0,0), # (0, 0.0174533),
                    qp_solver='PARTIAL_CONDENSING_HPIPM',
                    qp_solver_iter_max=20,
                    nlp_solver_type='SQP', # 'SQP_RTI' or 'SQP'
                    nlp_solver_max_iter=40,
                    open_loop=False,
                 ):
        self.title = title
        self.Ts = Ts
        self.N = N
        self.T_tot = T_tot
        self.model_f = model_f
        self.tire_f = tire_f
        self.sim_model_f = sim_model_f
        self.sim_tire_f = sim_tire_f
        self.μ_err = μ_err
        self.stiff_k = stiff_k
        self.J_k = J_k
        self.x_eq = x_eq
        self.u_eq = u_eq
        self.delta_lb = delta_lb
        self.delta_ub = delta_ub
        self.Fx_lb = Fx_lb
        self.Fx_ub = Fx_ub
        self.x0 = x0
        self.u0 = u0
        self.mpc_ws = mpc_ws
        self.V_ref = piecewise_constant(V_ref[0], [T_tot*ki for ki in V_ref[1]], Ts)[0] if type(V_ref) is tuple else V_ref 
        self.beta_ref = piecewise_constant(beta_ref[0], [T_tot*ki for ki in beta_ref[1]], Ts)[0] if type(beta_ref) is tuple else beta_ref
        self.r_ref = piecewise_constant(r_ref[0], [T_tot*ki for ki in r_ref[1]], Ts)[0] if type(r_ref) is tuple else r_ref
        self.delta_ref = piecewise_constant(delta_ref[0], [T_tot*ki for ki in delta_ref[1]], Ts)[0] if type(delta_ref) is tuple else delta_ref
        self.Fx_ref = piecewise_constant(Fx_ref[0], [T_tot*ki for ki in Fx_ref[1]], Ts)[0] if type(Fx_ref) is tuple else Fx_ref
        assert self.V_ref.shape == self.beta_ref.shape == self.r_ref.shape == self.delta_ref.shape == self.Fx_ref.shape, \
            f'V_ref: {self.V_ref.shape}, beta_ref: {self.beta_ref.shape}, r_ref: {self.r_ref.shape}, delta_ref: {self.delta_ref.shape}, Fx_ref: {self.Fx_ref.shape} must have the same shape'
        self.shifting = shifting
        self.ref_preview = ref_preview
        self.V_noise = V_noise
        self.beta_noise = beta_noise
        self.qp_solver = qp_solver
        self.qp_solver_iter_max = qp_solver_iter_max
        self.nlp_solver_type = nlp_solver_type
        self.nlp_solver_max_iter = nlp_solver_max_iter
        self.open_loop = open_loop


## PARAMETERS
GO_FAST = False # save figs instead of videos to run tests faster
# GO_FAST = True # save figs instead of videos to run tests faster

x_eq0 = [4.510306923563673, -0.4363323129985824, 1.503435641187891, -0.13754984841368398, 45.87960340367423]

x0_standstill = [1.0, -0.01, 0.01, 0.0, 0.0] # initial condition for standstill tests

# tracking tests
tt_tot = 15.0 # total time for tracking tests
tTs = 0.01 # sampling time for tracking tests
tzero_ref = piecewise_constant([[0]],[tt_tot], tTs)[0]
ttime_vector = np.linspace(0, tt_tot, len(tzero_ref))

# sinusoidal beta reference
beta_ref0 = (-22 + 8*np.sin(3*2*π*ttime_vector/tt_tot)).reshape(-1,1)*π/180 
beta_ref1 = (25*np.sin(2*2*π*ttime_vector/tt_tot)).reshape(-1,1)*π/180 

# sinusoidal V reference
V_ref0 = (5 + 1.5*np.sin(2*2*π*ttime_vector/tt_tot)).reshape(-1,1) # 4.5 m/s + 0.5 m/s sinusoidal

TESTS = [
    # Test(title='Open loop from Equilibrium, same model',
    #     T_tot=3.0, x_eq=x_eq0, x0=x_eq0, open_loop=True),
    # Test(title='Open loop from Equilibrium, model mismatch',
    #     sim_tire_f=pacejka_ca,
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     T_tot=3.0, x_eq=x_eq0, x0=x_eq0, open_loop=True),
    # Test(title='MPC from Equilibrium, model mismatch, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]',
    #     sim_tire_f=pacejka_ca, 
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=3.0, x_eq=x_eq0, x0=x_eq0),
    # Test(title='Open loop from Equilibrium, 1% IC deviation',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     T_tot=3.0, x_eq=x_eq0, x0=[x*0.99 for x in x_eq0], open_loop=True),
    # Test(title='MPC from Equilibrium, 1% IC deviation, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=3.0, x_eq=x_eq0, x0=[x*0.99 for x in x_eq0]),
    Test(title='MPC from Equilibrium, 30% IC deviation, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]',
        V_ref=([x_eq0[0]], [1]), 
        beta_ref=([x_eq0[1]], [1]), 
        r_ref=([x_eq0[2]], [1]), 
        delta_ref=([x_eq0[3]], [1]), 
        Fx_ref=([x_eq0[4]], [1]),
        mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
        T_tot=6.0, x_eq=x_eq0, x0=[x*1.3 for x in x_eq0]),
    # Test(title='MPC from Equilibrium, weight only β, w:[0, 1e3, 0, 0, 0, 1e1, 1e-2]',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[0, 1e3, 0, 0, 0, 1e1, 1e-2],
    #     T_tot=12.0, x_eq=x_eq0, x0=[x*1.3 for x in x_eq0]),
    # Test(title='MPC from Equilibrium, -30% IC, weight only V and r, w:[1e2, 0, 1e2, 0, 0, 1e1, 1e-2]',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[1e2, 0, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=[x*0.7 for x in x_eq0]),
    # Test(title='MPC from Equilibrium, +30% IC, weight only V and r, w:[1e2, 0, 1e2, 0, 0, 1e1, 1e-2]',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[1e2, 0, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=[x*1.3 for x in x_eq0]),
    # Test(title='MPC from standstill, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=x0_standstill),
    # Test(title='MPC from standstill, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]\nwith model mismatch',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     sim_model_f=DTM_model_dt_inputs_sim, 
    #     sim_tire_f=pacejka_ca, 
    #     μ_err=-0.05, 
    #     stiff_k=1.05, 
    #     J_k=0.9,
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=x0_standstill),
    # Test(title='MPC from standstill, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     sim_model_f=DTM_model_dt_inputs_sim, 
    #     sim_tire_f=pacejka_ca, 
    #     μ_err=-0.05, 
    #     stiff_k=1.05, 
    #     J_k=0.9,
    #     V_noise=(0,0.3), # (0, 0.3), #
    #     beta_noise=(0,0.018), # (0, 0.0174533),
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=x0_standstill,
    #     qp_solver_iter_max=20,
    #     nlp_solver_type='SQP', # 'SQP_RTI' or 'SQP'
    #     nlp_solver_max_iter=40),
    # Test(title='MPC from standstill, w:[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise, SQP_RTI',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=([x_eq0[1]], [1]), 
    #     r_ref=([x_eq0[2]], [1]), 
    #     delta_ref=([x_eq0[3]], [1]), 
    #     Fx_ref=([x_eq0[4]], [1]),
    #     sim_model_f=DTM_model_dt_inputs_sim, 
    #     sim_tire_f=pacejka_ca, 
    #     μ_err=-0.05, 
    #     stiff_k=1.05, 
    #     J_k=0.9,
    #     V_noise=(0,0.3), # (0, 0.3), #
    #     beta_noise=(0,0.018), # (0, 0.0174533),
    #     mpc_ws=[1e2, 1e3, 1e2, 0, 0, 1e1, 1e-2],
    #     T_tot=6.0, x_eq=x_eq0, x0=x0_standstill,
    #     qp_solver_iter_max=5,
    #     nlp_solver_max_iter=5,
    #     nlp_solver_type='SQP_RTI'), # 'SQP_RTI' or 'SQP'
    # Test(title='Sinusoidal β reference 1, w:[1e2, 1e3, 0, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise, SQP_RTI',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=beta_ref0, 
    #     r_ref=([0], [1]), 
    #     delta_ref=([0], [1]), 
    #     Fx_ref=([0], [1]),
    #     sim_model_f=DTM_model_dt_inputs_sim, 
    #     sim_tire_f=pacejka_ca, 
    #     μ_err=-0.05, 
    #     stiff_k=1.05, 
    #     J_k=0.9,
    #     V_noise=(0,0.3), # (0, 0.3), #
    #     beta_noise=(0,0.018), # (0, 0.0174533),
    #     mpc_ws=[1e2, 1e3, 0, 0, 0, 1e1, 1e-2],
    #     T_tot=tt_tot, x_eq=x_eq0, x0=x0_standstill, Ts=tTs,
    #     qp_solver_iter_max=5,
    #     nlp_solver_max_iter=5,
    #     nlp_solver_type='SQP_RTI',
    #     ),
    # Test(title='Sinusoidal β reference 2, w:[1e2, 1e3, 0, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise, SQP_RTI',
    #     V_ref=([x_eq0[0]], [1]), 
    #     beta_ref=beta_ref1, 
    #     r_ref=([0], [1]), 
    #     delta_ref=([0], [1]), 
    #     Fx_ref=([0], [1]),
    #     sim_model_f=DTM_model_dt_inputs_sim, 
    #     sim_tire_f=pacejka_ca, 
    #     μ_err=-0.05, 
    #     stiff_k=1.05, 
    #     J_k=0.9,
    #     V_noise=(0,0.3), # (0, 0.3), #
    #     beta_noise=(0,0.018), # (0, 0.0174533),
    #     mpc_ws=[1e2, 1e3, 0, 0, 0, 1e1, 1e-2],
    #     T_tot=tt_tot, x_eq=x_eq0, x0=x0_standstill, Ts=tTs,
    #     qp_solver_iter_max=5,
    #     nlp_solver_max_iter=5,
    #     nlp_solver_type='SQP_RTI',
    #     ),
    Test(title='Steps β and V reference, w:[1e2, 1e3, 0, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise, SQP_RTI',
        V_ref=([4, 7, 4], [1/3, 1/3, 1/3]),
        beta_ref=([-0.52, 0.52], [1/2, 1/2]),
        r_ref=([0], [1]), 
        delta_ref=([0], [1]), 
        Fx_ref=([0], [1]),
        sim_model_f=DTM_model_dt_inputs_sim, 
        sim_tire_f=pacejka_ca, 
        μ_err=-0.05, 
        stiff_k=1.05, 
        J_k=0.9,
        V_noise=(0,0.3), # (0, 0.3), #
        beta_noise=(0,0.018), # (0, 0.0174533),
        mpc_ws=[1e2, 1e3, 0, 0, 0, 1e1, 1e-2],
        T_tot=12.0, x_eq=x_eq0, x0=x0_standstill,
        qp_solver_iter_max=5,
        nlp_solver_max_iter=5,
        nlp_solver_type='SQP_RTI',
        # N=150, # increase N for longer prediction horizon
        ),
    Test(title='Steps β and V reference, w:[1e2, 1e3, 0, 0, 0, 1e1, 1e-2]\nwith model mismatch and measurement noise, SQP_RTI',
        V_ref=([7, 4, 7], [1/3, 1/3, 1/3]),
        beta_ref=([-0.52, 0.52, -0.52], [1/3, 1/3, 1/3]),
        r_ref=([0], [1]), 
        delta_ref=([0], [1]), 
        Fx_ref=([0], [1]),
        sim_model_f=DTM_model_dt_inputs_sim, 
        sim_tire_f=pacejka_ca, 
        μ_err=-0.05, 
        stiff_k=1.05, 
        J_k=0.9,
        V_noise=(0,0.3), # (0, 0.3), #
        beta_noise=(0,0.018), # (0, 0.0174533),
        mpc_ws=[1e2, 1e3, 0, 0, 0, 1e1, 1e-2],
        T_tot=12.0, x_eq=x_eq0, x0=x0_standstill,
        qp_solver_iter_max=5, 
        nlp_solver_max_iter=5,
        nlp_solver_type='SQP_RTI',
        # N=150, # increase N for longer prediction horizon
        ),
]

# create the generated_figures directory and make it empty
if not os.path.exists('generated_figures'):
    os.makedirs('generated_figures')
else:
    for f in os.listdir('generated_figures'):
        file_path = os.path.join('generated_figures', f)
        if os.path.isfile(file_path): os.remove(file_path)




n_test = 0
while n_test < len(TESTS): # repeat tests until they all converge
    test = TESTS[n_test]
    print(f"Test {n_test+1}/{len(TESTS)}: {test.title}")

    clear_previous_simulation()
    # define simulation fundamental time step [s]
    ts_sim = 0.001

    # setup controller parameters
    # - system model
    model = test.model_f(tire=test.tire_f, μ_err=0.0, stiff_k=1.0, J_k=1.0)
    # - controller sample time [s]
    Ts = test.Ts
    # - number of shooting time intervals 
    N  = test.N #50
    # - prediction horizon length [s]
    T = N*Ts

    T_tot = test.T_tot #12.9 #10.0 # total simulation time [s]

    # model used to simulate the system
    sim_model = test.sim_model_f(tire=test.sim_tire_f, μ_err=test.μ_err, stiff_k=test.stiff_k, J_k=test.J_k)

    # Equilibrium point (found in PYTHON)
    # x_eq, u_eq = [3.610849747542315, -0.4363323129985824, 1.2036165825141052, -0.18825862766328222, 27.47665205296075], [0.0, 0.0] 
    # x_eq, u_eq = [4.486209860862883, -0.4363323129985824, 1.4954032869542941, -0.11596738898598893, 46.64426852037662], [0.0, 0.0]
    x_eq = np.array(test.x_eq) # [m/s, rad, rad/s, rad, N]
    u_eq = np.array(test.u_eq) # [rad/s, N/s]

    V_eq = x_eq[0] # [m/s] total velocity
    beta_eq = x_eq[1] # [rad] sideslip angle
    r_eq = x_eq[2] # [rad/s] yaw rate
    delta_eq = u_eq[0] # [rad] steering angle 
    Fx_eq = u_eq[1] # [N] longitudinal force

    # Input bounds
    delta_lb = test.delta_lb # lower bound on steering angle
    delta_ub = test.delta_ub  # upper bound on steering angle
    Fx_lb = test.Fx_lb # lower bound on longitudinal force
    Fx_ub = test.Fx_ub # upper bound of longitudinal force

    # State bounds
    V_lb = MIN_V # [m/s] lower bound on velocity
    V_ub = MAX_V # [m/s] upper bound on velocity

    # initial condition
    x0 = np.array(test.x0) # [m/s, rad, rad/s, rad, N]
    u0 = np.array(test.u0) # [rad/s, N/s]

    # define cost weigth matrices
    w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = test.mpc_ws


    Q = np.diag([w_V, w_beta, w_r, w_delta, w_Fx])
    R = np.diag([w_dt_delta, w_dt_Fx])


    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()

    zero_ref, Tf = piecewise_constant([[0]],[T_tot], Ts)
    # V_ref, _    = piecewise_constant(test.V_ref[0], [T_tot*ki for ki in test.V_ref[1]], Ts)
    # beta_ref, _  = piecewise_constant(test.beta_ref[0], [T_tot*ki for ki in test.beta_ref[1]], Ts)
    # r_ref, _     = piecewise_constant(test.r_ref[0], [T_tot*ki for ki in test.r_ref[1]], Ts)
    # delta_ref, _ = piecewise_constant(test.delta_ref[0], [T_tot*ki for ki in test.delta_ref[1]], Ts)
    # Fx_ref, _    = piecewise_constant(test.Fx_ref[0], [T_tot*ki for ki in test.Fx_ref[1]], Ts)
    V_ref, beta_ref, r_ref, delta_ref, Fx_ref = test.V_ref, test.beta_ref, test.r_ref, test.delta_ref, test.Fx_ref

    # - provide a reference for all variables
    y_ref_nolookahead = np.column_stack((V_ref, beta_ref, r_ref, delta_ref, Fx_ref, zero_ref, zero_ref))

    # - add N samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref_nolookahead, np.repeat(y_ref_nolookahead[-1].reshape(1,-1), N, axis=0)))

    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

    # configure whether to apply shifting and to enable reference look-ahead
    shifting    = test.shifting
    ref_preview = test.ref_preview

    def create_ocp_solver_description(model, N, T, x0, yref) -> AcadosOcp:

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # define system dynamics model
        ocp.model = model

        # set prediction horizon:
        # tf - prediction horizon length [s]
        # N  - number of intervals in which the prediction horizon is divided 
        ocp.solver_options.tf = T
        # ocp.dims.N = N # deprecated
        ocp.solver_options.N_horizon = N
        
        # get state, control and cost dimensions
        nx = model.x.rows()
        nu = model.u.rows()

        ny = nx + nu 
        ny_e = nx

        # print(f"nx: {nx}, nu: {nu}, ny: {ny}, ny_e: {ny_e}")
    
        # define cost type
        # ocp.cost.cost_type = 'LINEAR_LS'
        # ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # print(f'Q: \n{Q}\nR: \n{R}')
        ocp.cost.W = block_diag(Q, R)
        ocp.cost.W_e = T/N*Q

        # # define matrices characterizing the cost
        # ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
        # ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
        # ocp.cost.Vx_e = np.eye(nx)
        
        # alternatively, for the NONLINEAR_LS cost type
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        # initialize variables for reference
        ocp.cost.yref = yref[0,:]
        ocp.cost.yref_e = yref[0,:nx]

        # # set constraints
        # ocp.constraints.lbu = np.array([delta_lb, Fx_lb])  # lower bounds on control inputs
        # ocp.constraints.ubu = np.array([delta_ub, Fx_ub])  # upper bounds on control inputs
        # ocp.constraints.idxbu = np.array([0, 1])  # indices of control inputs in the control vector

        ocp.constraints.lbx = np.array([delta_lb, Fx_lb])
        ocp.constraints.ubx = np.array([delta_ub, Fx_ub])
        ocp.constraints.idxbx = np.array([3, 4])

        # initialize constraint on initial condition
        ocp.constraints.x0 = x0

        # set solver options
        # ocp.solver_options.print_level = 1
        ocp.solver_options.qp_solver = test.qp_solver #"PARTIAL_CONDENSING_HPIPM"  #FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = test.nlp_solver_type # "SQP_RTI" #SQP, SQP_RTI

        # to configure partial condensing
        #ocp.solver_options.qp_solver_cond_N = int(N/10)

        # some more advanced settings (refer to the documentation to see them all)
        # - maximum number of SQP iterations (default: 100)
        ocp.solver_options.nlp_solver_max_iter = test.nlp_solver_max_iter #50
        # - maximum number of iterations for the QP solver (default: 50)
        ocp.solver_options.qp_solver_iter_max = test.qp_solver_iter_max #25

        # - configure warm start of the QP solver (0: no, 1: warm start, 2: hot start)
        # (depends on the specific solver)
        ocp.solver_options.qp_solver_warm_start = 0
        
        return ocp

    # setup simulation of system dynamics
    sim = AcadosSim()
    sim.model = sim_model
    sim.solver_options.T = ts_sim
    sim.solver_options.integrator_type = 'ERK'
    sim.generate_external_functions() # to hopefully recompile code

    acados_integrator = AcadosSimSolver(sim, verbose=False)

    # create OCP solver
    ocp = create_ocp_solver_description(model, N, T, x0, yref=y_ref)
    acados_ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    # initialize solver
    for stage in range(N):
        acados_ocp_solver.set(stage, "x", x0)
        acados_ocp_solver.set(stage, "u", u0)

    acados_ocp_solver.set(N, "x", x0)

    # create variables to store state and control trajectories 
    simX = np.zeros((N_steps + 1, nx))
    simU = np.zeros((N_steps_dt, nu))
    # set intial state
    simX[0, :] = x0

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N+1, nx, N_steps_dt + 1))
    x_opt[:, :, 0] = np.repeat(x0.reshape(1,-1),N+1, axis=0)
    u_opt = np.zeros((N, nu, N_steps_dt + 1))

    # variable to store total CPU time
    cpt = np.zeros((N_steps_dt,))

    # do some initial iterations to start with a good initial guess
    if not test.open_loop: 
        for _ in range(5): acados_ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

    # simulation loop
    k = 0 # iteration counter fo control loop
    for i in tqdm(range(N_steps), desc=f"Test {n_test+1}", ascii=False, ncols=75, colour='yellow'):

        # check whether to update the discrete-time part of the loop
        if(i % n_update == 0):
            if not test.open_loop: # skip updating the discrete-time part if in open loop mode
                # update reference
                for j in range(N): acados_ocp_solver.set(j, "yref", y_ref[k + (j if ref_preview else 0), :])
                acados_ocp_solver.set(N, "yref", y_ref[k + (N if ref_preview else 0), 0:-nu])

                
                # if performing shifting, explicitly initialize solver
                # (otherwise, it will be automatically intialized with the previous solution)
                if shifting:
                    for stage in range(N):
                        acados_ocp_solver.set(stage, "x", x_opt[stage+1, :, k])
                        acados_ocp_solver.set(stage, "u", u_opt[min([stage+1,N-1]), :, k])
                    acados_ocp_solver.set(N, "x", x_opt[N, :, k])
                
                
                # update the control 
                meas_state = simX[i,:].copy()
                # # meas_state[1] += np.random.normal(np.deg2rad(0), np.deg2rad(2)) # white noise on beta meas
                # meas_state[0] += np.random.normal(0, .3) # white noise on V meas
                # meas_state[1] += np.random.normal(0, np.deg2rad(1)) # white noise on beta meas

                meas_state[0] += np.random.normal(test.V_noise[0], test.V_noise[1]) # white noise on V meas
                meas_state[1] += np.random.normal(test.beta_noise[0], test.beta_noise[1]) # white noise on beta meas
            
                simU[k, :] = acados_ocp_solver.solve_for_x0(meas_state, fail_on_nonzero_status=False, print_stats_on_failure=False)
                # simU[k, :] = acados_ocp_solver.solve_for_x0(simX[i, :])

                # store CPU time required for solving the problem
                cpt[k] = acados_ocp_solver.get_stats('time_tot')

                # store optimal solution
                for stage in range(N):
                    x_opt[stage, :, k+1] = acados_ocp_solver.get(stage, "x")
                    u_opt[stage, :, k+1] = acados_ocp_solver.get(stage, "u")
                    
                x_opt[N, :, k+1] = acados_ocp_solver.get(N, "x")
            k += 1
            
        # simulate system
        if test.open_loop: simX[i + 1, :] = acados_integrator.simulate(simX[i,:], u0) # open loop test
        else: simX[i + 1, :] = acados_integrator.simulate(simX[i,:], simU[k-1, :])




    time        = np.linspace(0, ts_sim * N_steps, N_steps + 1)
    time_mpc    = np.linspace(0, Ts * (N_steps_dt-1), N_steps_dt+1)

    y_ref_plot = piecewise_constant(y_ref_nolookahead[:-1], [Ts]*(y_ref_nolookahead.shape[0]-1), ts_sim)[0]
    errors = np.abs(y_ref_plot[:,:3] - simX[:,:3])

    # plot the simulation results
    CM = 'jet' #'inferno'
    fig = plt.figure(figsize=(16, 9))
    plt.subplot(4,2,1)
    plt.plot(time, simX[:, 0], label='V')
    plt.plot(time_mpc, y_ref_nolookahead[:, 0], linestyle='--', label='V_ref')
    plt.title('Velocity, V')
    plt.xlabel('Time [s]')
    plt.ylabel('V [m/s]')
    plt.ylim(1.1*V_lb, 1.1*V_ub)
    plt.legend()

    plt.subplot(4,2,2)
    plt.plot(time, errors[:,0], label='error')
    plt.title('Error Velocity')
    plt.ylim(0, max(0.1, np.max(errors[:,0])))
    plt.xlabel('Time [s]')

    plt.subplot(4,2,3)
    plt.plot(time, np.rad2deg(simX[:, 1]), label='β')
    plt.plot(time_mpc, np.rad2deg(y_ref_nolookahead[:, 1]), linestyle='--', label='β_ref')
    plt.title('Sideslip Angle, β')
    plt.xlabel('Time [s]') 
    plt.ylabel('β [deg]')
    plt.ylim(-60, 60)
    plt.legend()

    plt.subplot(4,2,4)
    plt.plot(time, np.rad2deg(errors[:,1]), label='error')
    plt.title('Error Sideslip Angle')
    plt.ylim(0, max(0.1, np.max(np.rad2deg(errors[:,1]))))
    plt.xlabel('Time [s]')

    plt.subplot(4,2,5)
    plt.plot(time, np.rad2deg(simX[:, 2]), label='r')
    plt.plot(time_mpc, np.rad2deg(y_ref_nolookahead[:, 2]), linestyle='--', label='r_ref')
    plt.title('Yaw Rate, r')
    plt.xlabel('Time [s]')
    plt.ylabel('r [rad/s]')
    plt.ylim(np.rad2deg(-4),np.rad2deg(4))
    plt.legend()

    plt.subplot(4,2,6)
    plt.plot(time, np.rad2deg(errors[:,2]), label='error')
    plt.title('Error Yaw Rate')
    plt.ylim(0, max(0.1, np.max(np.rad2deg(errors[:,2]))))
    plt.xlabel('Time [s]')

    plt.subplot(4,2,7)
    plt.plot(time, np.rad2deg(simX[:, 3]), label='δ')
    plt.plot(time_mpc, np.rad2deg(y_ref_nolookahead[:,3]), linestyle='--', label='δ_ref')
    plt.title('Steering angle (at the wheel), δ')
    plt.xlabel('Time [s]')
    plt.ylabel('δ [deg]')
    plt.ylim(np.rad2deg(1.1*delta_lb), np.rad2deg(1.1*delta_ub))
    plt.legend()

    plt.subplot(4,2,8)
    plt.plot(time, simX[:, 4], label='Fx')
    plt.plot(time_mpc, y_ref_nolookahead[:,4], linestyle='--', label='Fx_ref')
    plt.title('Rear wheel longitudinal force, Fx')
    plt.xlabel('Time [s]')
    plt.ylabel('Fx [N]')
    plt.ylim(1.1*Fx_lb, 1.1*Fx_ub)
    plt.legend()


    plt.suptitle(f'{test.title}', fontsize=16)
    plt.tight_layout()

    save_title = f"{n_test+1}_{test.title.replace(' ', '_').replace('/', '_').replace('-', '_').replace(',', '_').replace('.', '_').replace('(', '').replace(')', '').replace(':', '').replace('\n', '_')}"
    plt.savefig(f'generated_figures/{save_title}.png', dpi=300, bbox_inches='tight')

    # plt.show()


    print(f"Final state: V={simX[-1,0]:.2f} m/s, Beta={np.rad2deg(simX[-1,1]):.2f} deg, Yaw rate={np.rad2deg(simX[-1,2]):.2f} deg/s, Delta={np.rad2deg(simX[-1,3]):.2f} deg, Fx={simX[-1,4]:.2f} N")


    # animation
    save_fps = 30.0
    # save_fps = 20.0
    try:
        anim = car_anim(
            xs=simX[:,:3],  # use the state vector as input
            # us=piecewise_constant(simU, [Ts]*simU.shape[0], ts_sim)[0],
            us=piecewise_constant(simX[:-1,3:5], [ts_sim]*(simX.shape[0]-1), ts_sim)[0],
            ic=np.array([0, 0, π/2]),  # initial conditions (x, y, ψ) 
            dt=ts_sim,  # time step
            fps=save_fps,  # frames per second
            speed=2.0,  # speed factor for the animation 
            # follow=True,  # follow the car in the animation
            title=test.title,  # title of the animation
            # get_video=True,  # get video instead of jshtml
            static_img=GO_FAST,  # use static image instead of animation
            no_notebook=True,  # do not use notebook mode
        )  # run the car animation with the STM results

        if GO_FAST:
            plt.savefig(f'generated_figures/{save_title}_anim.png', dpi=300, bbox_inches='tight')
        else:
            anim.save(f'generated_figures/{save_title}.mp4', fps=save_fps, extra_args=['-vcodec', 'libx264']) # save animation as mp4
            # convert to gif using ffmpeg command
            os.system(f'ffmpeg -loglevel error -y -i generated_figures/{save_title}.mp4 -vf "fps={save_fps},scale=1920:-1:flags=lanczos" generated_figures/{save_title}.gif')
            # os.system(f'ffmpeg -loglevel error -y -i generated_figures/{save_title}.mp4 -vf "fps=60,scale=1080:-1:flags=lanczos" generated_figures/{save_title}_60fps.gif')
        n_test += 1  # increment test number for the next test 
    
    except Exception as e: 
        print(f'Failed test: {test.title} -> {e}. Retrying...')

    plt.close('all')
# plt.show()



