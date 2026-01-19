from mg_utils import *

### MPC PARAMETERS ###
## Constraints
print(f"Max sideslip angle set to {np.rad2deg(MAX_BETA):.2f} deg")
LBX, UBX, IDXBX = [-MAX_BETA, -MAX_DELTA, MIN_FX], [MAX_BETA, MAX_DELTA, MAX_FX], [1,3,4] # lower bounds on states
# LBU, UBU, IDXBU = [-MAX_D_DELTA, -MAX_D_FX], [MAX_D_DELTA, MAX_D_FX], [0,1] # both input boundeed
LBU, UBU, IDXBU = [-MAX_D_DELTA], [MAX_D_DELTA], [0] # delta only
# LBU, UBU, IDXBU = [], [], [] # no bounds on inputs

# define cost weigth matrices
# w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = 1, 5, 0, 1e-1, 1e-5, 1e-1, 1e-5 # <-
# w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = 1, 5, 0, 3e-2, 1e-5, 3e-2, 1e-5 
w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = 1, 5, 0, 3e-3, 1e-6, 3e-2, 1e-5 
Q = np.diag([w_V, w_beta, w_r, w_delta, w_Fx])
R = np.diag([w_dt_delta, w_dt_Fx])

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
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  #FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM <-
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
    ocp.solver_options.qp_solver_warm_start = 0

    return ocp

class MPC_Controller():
    def __init__(self, model=STM_model_dt_inputs, N=100, T=1.0, hard_cold_start=False):
        self.model, self.N, self.T = model, N, T
        self.ocp = create_ocp_solver_description(model, N, T, Q, R, LBX, UBX, IDXBX, LBU, UBU, IDXBU)
        self.solv = AcadosOcpSolver(self.ocp, verbose=False) 

        # Save a clean initial guess to reset the solver at each time step
        self.hard_cold_start = hard_cold_start
        if hard_cold_start:
            self.reset_file = "cold_start_data.json"
            self.solv.store_iterate(self.reset_file, overwrite=True)

    def get_ctrl(self, x, y_ref):
        if self.hard_cold_start: self.solv.load_iterate(self.reset_file, verbose=False)
        for j in range(self.N): self.solv.set(j, "yref", y_ref)
        # print(f'x [{x.shape}]: {x}, y_ref [{y_ref.shape}]: {y_ref}')
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