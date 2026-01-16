from mg_utils import *

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
def create_ocp_solver_description(model, N, T, Q, R, lbx, ubx, idxbx, lbu, ubu, idxbu) -> AcadosOcp:
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

    # initialize constraint on initial condition, default intial state, v must be > 0
    x0 = np.zeros(nx); x0[0] = 1.0
    ocp.constraints.x0 = x0

    # set constraints
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = idxbx
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = idxbu

    # set solver options
    # ocp.solver_options.print_level = 1
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  #FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP, SQP_RTI

    # to configure partial condensing
    #ocp.solver_options.qp_solver_cond_N = int(N/10)

    # some more advanced settings (refer to the documentation to see them all)
    # - maximum number of SQP iterations (default: 100)
    ocp.solver_options.nlp_solver_max_iter = 20 #50 <-
    # - maximum number of iterations for the QP solver (default: 50)
    ocp.solver_options.qp_solver_iter_max = 5 #25 <-

    # - configure warm start of the QP solver (0: no, 1: warm start, 2: hot start)
    # (depends on the specific solver)
    ocp.solver_options.qp_solver_warm_start = 0

    return ocp

class MPC_Controller():
    def __init__(self, model, N, T, Q, R, lbx, ubx, idxbx, lbu, ubu, idxbu):
        self.model, self.N, self.T = model, N, T
        ocp = create_ocp_solver_description(model, N, T, Q, R, lbx, ubx, idxbx, lbu, ubu, idxbu) # define ocp
        self.solv = AcadosOcpSolver(ocp, verbose=False) # define solver

    def get_ctrl(self, x, y_ref):
        for j in range(self.N): self.solv.set(j, "yref", y_ref)
        u = self.solv.solve_for_x0(x, fail_on_nonzero_status=False, print_stats_on_failure=False)
        return u

    def get_stats(self):
        x_opt, u_opt = np.zeros((self.N+1, self.model.x.rows())), np.zeros((self.N, self.model.u.rows()))
        for s in range(self.N+1):
            x_opt[s, :] = self.solv.get(s, "x")
            if s < self.N: u_opt[s, :] = self.solv.get(s, "u")
        cpu_time = self.solv.get_stats("time_tot")
        cost = self.solv.get_cost()
        return x_opt, u_opt, cpu_time, cost

# simulation
class Simulator():
    def __init__(self, sim_model, ts_sim, integrator_type='ERK'):
        # setup simulation of system dynamics
        sim = AcadosSim()
        sim.model = sim_model
        sim.solver_options.T = ts_sim
        sim.solver_options.integrator_type = integrator_type
        self.acados_integrator = AcadosSimSolver(sim, verbose=False)
    def step(self, x, u):
        return self.acados_integrator.simulate(x, u)