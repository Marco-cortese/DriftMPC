from mc_utils_LT import *

# utils
def clear_previous_simulation():
    vars = globals().copy()
    for v in globals().copy():
        if(type(vars[v]) is AcadosSimSolver or type(vars[v]) is AcadosOcpSolver):
            del globals()[v]

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



# ### MPC
def create_ocp_solver_description(model, N, T, Q, R, lbx, ubx, idxbx) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # define system dynamics model
    ocp.model = model

    # set prediction horizon:
    # tf - prediction horizon length [s]
    # N  - number of intervals in which the prediction horizon is divided 
    ocp.solver_options.tf = T
    ocp.solver_options.N_horizon = N

    # get state, control and cost dimensions
    nx = model.x.rows()
    nu = model.u.rows()

    ny = nx + nu 
    ny_e = nx

    print(f"nx: {nx}, nu: {nu}, ny: {ny}, ny_e: {ny_e}")

    # define cost type
    # ocp.cost.cost_type = 'LINEAR_LS'
    # ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    print(f'Q: \n{Q}\nR: \n{R}')
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
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # # set constraints
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = idxbx

    # initialize constraint on initial condition
    ocp.constraints.x0 = np.zeros(nx)

    # set solver options
    # ocp.solver_options.print_level = 1
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  #FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI" #SQP, SQP_RTI

    # to configure partial condensing
    #ocp.solver_options.qp_solver_cond_N = int(N/10)

    # some more advanced settings (refer to the documentation to see them all)
    # - maximum number of SQP iterations (default: 100)
    ocp.solver_options.nlp_solver_max_iter = 50 #50 1 <-
    # - maximum number of iterations for the QP solver (default: 50)
    ocp.solver_options.qp_solver_iter_max = 25 #25 5 <-

    # - configure warm start of the QP solver (0: no, 1: warm start, 2: hot start)
    # (depends on the specific solver)
    ocp.solver_options.qp_solver_warm_start = 0

    return ocp


# def create_mpc_controller():
