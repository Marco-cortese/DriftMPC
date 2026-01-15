from mpc import *
from scipy.linalg import block_diag

# define simulation fundamental time step [s]
ts_sim = 0.001
# model used to simulate the system
sim_model = DTM_model_LT_dt_inputs_sim(ts_sim)
# sim_model = model = DTM_model_dt_inputs_sim()

# setup controller parameters
# - controller sampling time [s]
Ts = 0.01
# - number of shooting time intervals 
N  = 100 #50
# - prediction horizon length [s]
T = N*Ts
# - system model
# model = DTM_model_LT_dt_inputs_sim(Ts)
model = STM_model_dt_inputs_sim()

T_tot = 6 #10.0 # total simulation time [s]


# sim_model = DTM_model_dt_inputs_sim()

# Equilibrium point (found in PYTHON) [V, beta, r, delta, Fx]
# x_eq, u_eq = [3.610849747542315, -0.4363323129985824, 1.2036165825141052], [-0.18825862766328222, 27.47665205296075]
# x_eq, u_eq = [4.486209860862883, -0.4363323129985824, 1.4954032869542941], [-0.11596738898598893, 46.64426852037662]

# Input bounds
delta_lb = -MAX_DELTA # lower bound on steering angle
delta_ub = MAX_DELTA  # upper bound on steering angle
Fx_lb = MIN_FX # lower bound on longitudinal force
Fx_ub = MAX_FX # upper bound of longitudinal force

# State bounds
V_lb = MIN_V # [m/s] lower bound on velocity
V_ub = MAX_V # [m/s] upper bound on velocity

# get state and control dimensions
nx, nu = model.x.rows(), model.u.rows()
nx_sim, nu_sim = sim_model.x.rows(),  sim_model.u.rows()

# initial condition
x0 = np.concatenate(([3.0], np.zeros(nx - 1)))
# x0 = x0 + np.concatenate([[np.random.uniform(-0.5, 0.5)], [np.deg2rad(np.random.uniform(-15, 15))], [np.random.uniform(-1, 1)], np.zeros(nx - 3)])

# x0_sim = np.concatenate(([1.0], np.zeros(nx_sim - 1)))
x0_sim = np.concatenate([x0, [0]])

# initial guess for the control inputs
u0 = np.array([0, 0])

# define cost weigth matrices
# w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = 1e3, 5e4, 0, 0, 0, 1e1, 1e-2 # <-
w_V, w_beta, w_r, w_delta, w_Fx, w_dt_delta, w_dt_Fx = 1e1, 5e2, 0, 0, 0, 3e-2, 1e-4
Q = np.diag([w_V, w_beta, w_r, w_delta, w_Fx])
R = np.diag([w_dt_delta, w_dt_Fx])

# define reference trajectories
zero_ref, _ = piecewise_constant([[0]],[T_tot], Ts)
V_ref, Tf    = piecewise_constant([4.5],[T_tot], Ts)
# V_ref, Tf    = piecewise_constant([5, 3, 5],[T_tot/3, T_tot/3, T_tot/3], Ts)
# beta_ref, _  = piecewise_constant([np.deg2rad(-30)],[T_tot], Ts)
beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30) ],[T_tot/2, T_tot/2], Ts)
# beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30), np.deg2rad(-30) ],[T_tot/3, T_tot/3, T_tot/3], Ts)
r_ref, _     = piecewise_constant([0],[T_tot], Ts)

# - provide a reference for all variables
#  x = [V, beta, r, delta, Fx] u = [d_delta, d_Fx]
# y_ref = np.column_stack((np.zeros((len(angle_ref), 1)), angle_ref.reshape(-1,1), np.zeros((len(angle_ref), nx+nu-2))))
y_ref_nolookahead = np.column_stack((V_ref, beta_ref, np.tile(zero_ref, nx+nu-2)))
# y_ref_nolookahead = np.column_stack((V_ref, beta_ref, zero_ref, zero_ref, zero_ref, zero_ref, zero_ref, zero_ref))

# - add N samples at the end (replicas of the last sample) for reference look-ahead
y_ref = np.vstack((y_ref_nolookahead, np.repeat(y_ref_nolookahead[-1].reshape(1,-1), N, axis=0)))
# compute the number of steps for simulation
N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

# configure whether to apply shifting and to enable reference look-ahead
shifting    = False
ref_preview = False

print(f'yref: {y_ref.shape}, N_steps sim: {N_steps}, N_steps controller: {N_steps_dt}, n_update: {n_update}')

# ## SIMULATION
# In[ ]:

# setup simulation of system dynamics
sim = AcadosSim()
sim.model = sim_model
sim.solver_options.T = ts_sim
sim.solver_options.integrator_type = 'ERK'

acados_integrator = AcadosSimSolver(sim, verbose=False)

# create OCP solver
ocp = create_ocp_solver_description(model, N, T, Q, R, 
                                    lbx=np.array([delta_lb, Fx_lb]), ubx=np.array([delta_ub, Fx_ub]), idxbx=np.array([3,4]))
acados_ocp_solver = AcadosOcpSolver(ocp, verbose=False)

# initialize solver
for stage in range(N):
    acados_ocp_solver.set(stage, "x", x0)
    acados_ocp_solver.set(stage, "u", u0)

acados_ocp_solver.set(N, "x", x0)

# create variables to store state and control trajectories 
simX = np.zeros((N_steps + 1, nx_sim))
simU = np.zeros((N_steps_dt, nu_sim))
# set intial state
simX[0, :] = x0_sim

# create variables to store, at each iteration, previous optimal solution
x_opt = np.zeros((N+1, nx, N_steps_dt + 1))
x_opt[:, :, 0] = np.repeat(x0.reshape(1,-1),N+1, axis=0)
u_opt = np.zeros((N, nu, N_steps_dt + 1))

# variable to store total CPU time
cpt = np.zeros((N_steps_dt,))

# do some initial iterations to start with a good initial guess
for j in range(N): acados_ocp_solver.set(j, "yref", y_ref[0, :]) # update reference
acados_ocp_solver.set(N, "yref", y_ref[0, 0:-nu])
# for _ in range(5): acados_ocp_solver.solve_for_x0(x0)
for _ in range(15): acados_ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

# simulation loop
k = 0 # iteration counter fo control loop
for i in tqdm(range(N_steps), desc="Simulation", ascii=False, ncols=75, colour='yellow'):

    # check whether to update the discrete-time part of the loop
    if(i % n_update == 0):
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
        meas_state = simX[i,:nx].copy()
        # meas_state[1] += np.random.normal(np.deg2rad(0), np.deg2rad(2)) # white noise on beta meas
        # meas_state[0] += np.random.normal(0, .3) # white noise on velocity meas

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
    simX[i + 1, :] = acados_integrator.simulate(simX[i,:], simU[k-1, :])
    # simX[i + 1, :] = acados_integrator.simulate(simX[i,:], u0) # open loop test


# ## PLOTS
# In[ ]:

time        = np.linspace(0, ts_sim * N_steps, N_steps + 1)
time_mpc    = np.linspace(0, Ts * (N_steps_dt-1), N_steps_dt+1)

y_ref_plot = piecewise_constant(y_ref_nolookahead[:-1], [Ts]*(y_ref_nolookahead.shape[0]-1), ts_sim)[0]
errors = y_ref_plot[:,:3] - simX[:,:3]

# plot the simulation results
CM = 'jet' #'inferno'
fig = plt.figure(figsize=(12, 12))
plt.subplot(5,2,1)
plt.plot(time_mpc, y_ref_nolookahead[:, 0], label='V_ref')
plt.plot(time, simX[:, 0], linestyle='--', label='V')
plt.title('Total Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Total Velocity (m/s)')
plt.ylim(1.1*V_lb, 1.1*V_ub)
plt.legend()

plt.subplot(5,2,2)
plt.plot(time, errors[:,0], label='error')
plt.title('Error Velocity')
plt.xlabel('Time (s)')

plt.subplot(5,2,3)
plt.plot(time_mpc, np.rad2deg(y_ref_nolookahead[:, 1]), label='beta_ref')
plt.plot(time, np.rad2deg(simX[:, 1]), linestyle='--', label='beta')
plt.title('Sideslip Angle')
plt.xlabel('Time (s)')
plt.ylabel('Sideslip angle (deg)')
plt.ylim(-60, 60)
plt.legend()

plt.subplot(5,2,4)
plt.plot(time, np.rad2deg(errors[:,1]), label='error')
plt.title('Error Sideslip Angle')
plt.xlabel('Time (s)')

plt.subplot(5,2,5)
plt.plot(time_mpc, np.rad2deg(y_ref_nolookahead[:, 2]), label='r_ref')
plt.plot(time, np.rad2deg(simX[:, 2]), linestyle='--', label='r')
plt.title('Yaw rate')
plt.xlabel('Time (s)')
plt.ylabel('Yaw rate (rad/s)')
plt.ylim(np.rad2deg(-4),np.rad2deg(4))
plt.legend()

plt.subplot(5,2,6)
plt.plot(time, np.rad2deg(errors[:,2]), label='error')
plt.title('Error Yaw Rate')
plt.xlabel('Time (s)')

plt.subplot(5,2,7)
plt.plot(time, np.rad2deg(simX[:, 3]), linestyle='--', label='delta')
plt.title('Steering angle (at the ground)')
plt.xlabel('Time (s)')
plt.ylabel('Steering angle (deg)')
plt.ylim(np.rad2deg(1.1*delta_lb), np.rad2deg(1.1*delta_ub))
plt.legend()

plt.subplot(5,2,8)
plt.plot(time, simX[:, 4], linestyle='--', label='Fx')
plt.title('Rear wheel longitudinal force')
plt.xlabel('Time (s)')
plt.ylabel('Longitudinal Force (N)')
plt.ylim(1.1*Fx_lb, 1.1*Fx_ub)
plt.legend()

plt.subplot(5,2,9)
plt.plot(time, Fz_Front_ST - simX[:, 5], label='Fz front')
plt.plot(time, Fz_Rear_ST + simX[:, 5], linestyle='--', label='Fz rear')
plt.title('Normal Force at the axis')
plt.xlabel('Time (s)')
plt.ylabel('Nominal Force (N)')
# plt.ylim(1.1*Fx_lb, 1.1*Fx_ub)
plt.legend()

plt.subplot(5,2,10)
plt.plot(time, simX[:, 5]*l/(m*h), label='ax')
plt.title('Longitudinal acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Longitudinal acceleration (m/s^2)')
# plt.ylim(1.1*Fx_lb, 1.1*Fx_ub)
plt.legend()


plt.suptitle('MPC simulation results', fontsize=16)
plt.tight_layout()
plt.show()


print("Final state:")
print(f"V: {simX[-1,0]:.2f} m/s")
print(f"Beta: {np.rad2deg(simX[-1,1]):.2f} deg")
print(f"Yaw rate: {np.rad2deg(simX[-1,2]):.2f} deg/s")
print(f"Delta: {np.rad2deg(simX[-1,3]):.2f} deg")
print(f"Fx: {simX[-1,4]:.2f} N")