from mpc import *
from numpy.random import uniform as unf
np.random.seed(0)  # for reproducibility

N_SIMULATIONS = 15  # number of simulations with different random initial conditions
ZERO_MPC_COST_THRESHOLD = 0.2  # threshold to consider cost as zero 

# random ic
v_lims = (2.0, 8.0)  # m/s # ficed limits
# v_lims = (YREF[0]-1, YREF[0]+1)  # m/s # around ref
beta_lims = (-40*π/180, 40*π/180)
r_lims = (0, 60*π/180)

# setup controller parameters
Ts = 0.01 # - controller sampling time [s]
N  = 100 #100 # - number of shooting time intervals e
T = N*Ts # - prediction horizon length [s]
T_tot = 2.5 #10.0 # total simulation time [s]

# ts_sim = 0.001 # simulation fundamental time step [s] 
ts_sim = Ts # simulation fundamental time step [s] = Ts

all_x0s, all_opt_us = [], []



# - system model
model = STM_model_dt_inputs()
# model = DTM_model_dt_inputs()
# model = DTM_model_LT_dt_inputs(Ts)

# - simulation model
sim_model = model  # use the same model for simulation

# setup simulation of system dynamics
sim = Simulator(sim_model=sim_model, ts_sim=ts_sim, integrator_type='ERK')

# set up MPC
mpc_ctrl = MPC_Controller(model, N, T, verbose=False)


for sample_idx in range(N_SIMULATIONS):  # run multiple simulations with different random initial conditions
    print(f"{'='*50}\nSimulation {sample_idx+1}\n{'='*50}")

    X0 = np.array([unf(*v_lims), unf(*beta_lims), unf(*r_lims), 0, 0])
    x0 = X0
    x0_sim = x0  # use the same initial condition for simulation

    print(f"Initial condition: V={X0[0]:.2f} m/s, beta={np.rad2deg(X0[1]):.2f} deg, r={np.rad2deg(X0[2]):.2f} deg/s, delta={np.rad2deg(X0[3]):.2f} deg, Fx={X0[4]:.2f} N")



    # # set up MPC
    # mpc_ctrl = MPC_Controller(model, N, T, verbose=False)
    # reset controller
    mpc_ctrl.reset()

    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()
    nx_sim, nu_sim = sim_model.x.rows(),  sim_model.u.rows()

    # define reference trajectories
    V_ref, Tf    = piecewise_constant([4.5],[T_tot], Ts)
    # V_ref, Tf    = piecewise_constant([5, 3, 5],[T_tot/3, T_tot/3, T_tot/3], Ts)
    beta_ref, _  = piecewise_constant([np.deg2rad(-30)],[T_tot], Ts)
    # beta_ref, _  = piecewise_constant([np.deg2rad(-20)],[T_tot], Ts)
    # beta_ref, _  = piecewise_constant([np.deg2rad(-40)],[T_tot], Ts)
    # beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30) ],[T_tot/2, T_tot/2], Ts)
    # beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30), np.deg2rad(-30) ],[T_tot/3, T_tot/3, T_tot/3], Ts)
    r_ref, _     = piecewise_constant([0],[T_tot], Ts)

    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)


    ## SIMULATION

    # create variables to store state and control trajectories 
    simX = np.zeros((N_steps + 1, nx_sim))
    simU = np.zeros((N_steps_dt, nu_sim))
    # set intial state
    simX[0, :] = x0_sim

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N+1, nx, N_steps_dt + 1))
    x_opt[:, :, 0] = np.repeat(x0.reshape(1,-1),N+1, axis=0)
    u_opt = np.zeros((N, nu, N_steps_dt + 1))

    # variable to store total CPU time and cost at each MPC iteration
    cpt, costs = np.zeros((N_steps_dt,)), np.zeros((N_steps_dt,))


    # simulation loop
    k = 0 # iteration counter fo control loop
    iterator = tqdm(range(N_steps), desc="Simulation", ascii=False, ncols=75, colour='yellow', mininterval=1/60, miniters=1)
    # iterator = range(N_steps)
    for i in iterator:
        if(i % n_update == 0): # check whether to update the discrete-time part of the loop
            yr = np.zeros(nx+nu); yr[:3] = V_ref[k,0], beta_ref[k,0], r_ref[k,0] # set reference trajectory
            simU[k, :] = mpc_ctrl.get_ctrl(simX[i,:nx], yr) # get control action
            x_opt[:,:,k+1], u_opt[:,:,k+1], cpt[k], costs[k] = mpc_ctrl.get_stats() # get stats        
            k += 1
        simX[i+1, :] = sim.step(simX[i,:], simU[k-1,:]) # simulate system dynamics

    # print("Average MPC computation time: %.4f ms" % (1e3*np.mean(cpt)))
    # print("Average MPC cost: %.4f " % (np.mean(costs)))

    # to filter out bad simulations: find the time idxs where mpc cost is < ZERO_MPC_COST_THRESHOLD and monotonically decreasing 
    assert np.all(costs >= 0), "MPC costs contain negative values!"
    zero_cost_idxs = np.where(costs < ZERO_MPC_COST_THRESHOLD)[0] # negligible costs
    zero_cost_mask = np.isin(np.arange(len(costs)), zero_cost_idxs)
    d_cost = np.diff(costs)
    first_t_idx = N # find index from which cost is negligible or always decreasing
    while first_t_idx > 0 and (zero_cost_mask[first_t_idx-1] or d_cost[first_t_idx-2] < 0): first_t_idx -= 1
    first_t_idx+=2 # to have some margin
    save_idxs = np.arange(first_t_idx, first_t_idx+N)  # only keep first window after first_t_idx
    print(f'first_t_id: {first_t_idx}')

    # save
    x_save = simX[save_idxs*n_update, :nx]  # downsample to controller rate
    u_save = simU[save_idxs, :]
    x_save_opt = x_opt[:N, :nx, first_t_idx+1]
    u_save_opt = u_opt[:N, :nu, first_t_idx+1]
    t_save = ((save_idxs) * ts_sim).reshape(-1,1)

    # print all shapes
    print(f"x_save shape: {x_save.shape}, x_save_opt shape: {x_save_opt.shape}, u_save shape: {u_save.shape}, u_save_opt shape: {u_save_opt.shape}, t_save shape: {t_save.shape}")


    ## PLOTS

    time        = np.linspace(0, ts_sim * N_steps, N_steps + 1)
    time_mpc    = np.linspace(0, Ts * (N_steps_dt-1), N_steps_dt+1)

    sim_V_ref = piecewise_constant(V_ref[:-1], [Ts]*(len(V_ref)-1), ts_sim)[0]
    sim_beta_ref = piecewise_constant(beta_ref[:-1], [Ts]*(len(beta_ref)-1), ts_sim)[0]
    sim_r_ref = piecewise_constant(r_ref[:-1], [Ts]*(len(r_ref)-1), ts_sim)[0]
    errors = np.column_stack((sim_V_ref, sim_beta_ref, sim_r_ref)) - simX[:,:3]

    if True: # plot
    # if False:
        # plot the simulation results
        CM = 'jet' #'inferno'
        fig = plt.figure(figsize=(20, 12))
        plt.subplot(5,2,1) # Velocity plots
        plt.plot(time, simX[:, 0], label='V')
        plt.plot(time_mpc, V_ref, linestyle=':', label='V_ref')
        plt.ylim(1.1*-MAX_V, 1.1*MAX_V)
        plt.title('Total Velocity [m/s]'), plt.xlabel('Time [s]'), plt.ylabel('Total Velocity [m/s]'), plt.legend()
        plt.subplot(5,2,2)
        plt.plot(time, errors[:,0], label='error')
        plt.plot(time, np.zeros_like(time), linestyle=':')
        plt.title('Error Velocity [m/s]'), plt.xlabel('Time [s]'), plt.ylabel('Error Velocity [m/s]')

        plt.subplot(5,2,3) # Sideslip angle plots
        plt.plot(time, np.rad2deg(simX[:, 1]), label='beta')
        plt.plot(time_mpc, np.rad2deg(beta_ref), linestyle=':', label='beta_ref')
        plt.ylim(-60, 60)
        plt.title('Sideslip Angle [deg]'), plt.xlabel('Time [s]'), plt.ylabel('Sideslip angle [deg]'), plt.legend()
        plt.subplot(5,2,4)
        plt.plot(time, np.rad2deg(errors[:,1]), label='error')
        plt.plot(time, np.zeros_like(time), linestyle=':')
        plt.title('Error Sideslip Angle [deg]'), plt.xlabel('Time [s]'), plt.ylabel('Error Sideslip Angle [deg]')

        # plt.subplot(5,2,5) # Yaw rate plots
        # plt.plot(time, np.rad2deg(simX[:, 2]), label='r')
        # plt.plot(time_mpc, np.rad2deg(r_ref), linestyle=':', label='r_ref')
        # plt.ylim(np.rad2deg(-4),np.rad2deg(4))
        # plt.title('Yaw rate'), plt.xlabel('Time (s)'), plt.ylabel('Yaw rate (rad/s)'), plt.legend()
        # plt.subplot(5,2,6)
        # plt.plot(time, np.rad2deg(errors[:,2]), label='error')
        # plt.plot(time, np.zeros_like(time), linestyle=':')
        # plt.title('Error Yaw Rate'), plt.xlabel('Time (s)'), plt.ylabel('Error Yaw Rate (deg)')

        # MPC control input plots
        plt.subplot(5,2,5) # derivatives of steering angle and longitudinal force (inputs)
        plt.plot(time[:-1], np.rad2deg(simU[:, 0]), label='d_delta')
        plt.plot(t_save, np.rad2deg(u_save[:,0]), label='d_delta (saved)', linestyle='--')
        plt.plot(t_save, np.rad2deg(u_save_opt[:,0]), label='d_delta (opt)', linestyle='--')
        plt.ylim(np.rad2deg(1.1*-MAX_D_DELTA), np.rad2deg(1.1*MAX_D_DELTA))
        plt.axhline(y=np.rad2deg(MAX_D_DELTA), color='gray', alpha=0.6), plt.axhline(y=-np.rad2deg(MAX_D_DELTA), color='gray', alpha=0.6)
        plt.title('Steering angle rate [deg/s]'), plt.xlabel('Time [s]'), plt.ylabel('Steering angle rate [deg/s]'), plt.legend(loc='upper right')

        plt.subplot(5,2,6) # longitudinal force rate
        plt.plot(time[:-1], simU[:, 1], label='d_Fx')
        plt.plot(t_save, u_save[:,1], label='d_Fx (saved)', linestyle='--')
        plt.plot(t_save, u_save_opt[:,1], label='d_Fx (opt)', linestyle='--')
        # plt.ylim(1.1*-MAX_D_FX, 1.1*MAX_D_FX)
        plt.axhline(y=MAX_D_FX, color='gray', alpha=0.6), plt.axhline(y=-MAX_D_FX, color='gray', alpha=0.6)
        plt.title('Rear wheel longitudinal force rate [N/s]'), plt.xlabel('Time [s]'), plt.ylabel('Longitudinal Force rate [N/s]'), plt.legend(loc='upper right')

        # control inputs plots / integration of actual inputs
        plt.subplot(5,2,7) # steering angle
        plt.plot(time, np.rad2deg(simX[:, 3]), label='delta')
        plt.plot(t_save, np.rad2deg(x_save[:,3]), label='delta (saved)', linestyle='--')
        plt.plot(t_save, np.rad2deg(x_save_opt[:,3]), label='delta (opt)', linestyle='--')
        # plt.ylim(np.rad2deg(1.1*-MAX_DELTA), np.rad2deg(1.1*MAX_DELTA))
        plt.axhline(y=np.rad2deg(MAX_DELTA), color='gray', alpha=0.6), plt.axhline(y=-np.rad2deg(MAX_DELTA), color='gray', alpha=0.6)
        plt.title('Steering angle [deg]'), plt.xlabel('Time [s]'), plt.ylabel('Steering angle [deg]'), plt.legend(loc='upper right')
        
        plt.subplot(5,2,8) # longitudinal force
        plt.plot(time, simX[:, 4], label='Fx')
        plt.plot(t_save, x_save[:,4], label='Fx (saved)', linestyle='--')
        plt.plot(t_save, x_save_opt[:,4], label='Fx (opt)', linestyle='--')
        plt.ylim(1.1*-MAX_FX, 1.1*MAX_FX)
        plt.axhline(y=MAX_FX, color='gray', alpha=0.6), plt.axhline(y=-MAX_FX, color='gray', alpha=0.6)
        plt.title('Rear wheel longitudinal force [N]'), plt.xlabel('Time [s]'), plt.ylabel('Longitudinal Force [N]'), plt.legend(loc='upper right')

        plt.subplot(5,2,9) # plot costs and CPU time
        plt.plot(time_mpc[:-1], costs, label='cost')
        plt.scatter(time_mpc[:-1][zero_cost_mask], costs[zero_cost_mask], label='zero cost', s=5, color='blue', zorder=5)
        plt.plot(time_mpc[:-2], -d_cost/np.max(np.abs(d_cost)), label='cost derivative')
        plt.axvline(x=time_mpc[first_t_idx], color='red', label='first valid idx')
        # plt.ylim(0, 1.1*max(costs))
        plt.title('MPC cost'), plt.xlabel('Time [s]'), plt.ylabel('Cost'), plt.legend(loc='upper right')

        plt.subplot(5,2,10)
        plt.plot(time_mpc[:-1], 1e3*cpt, label='CPU time')
        plt.ylim(0, 1.1*max(1e3*cpt))
        plt.title('MPC computation time [ms]'), plt.xlabel('Time [s]'), plt.ylabel('Computation time [ms]'), plt.legend(loc='upper right')

        # # load transfer plots
        # plt.subplot(5,2,9)
        # plt.plot(time, Fz_Front_ST - simX[:, 5], label='Fz front')
        # plt.plot(time, Fz_Rear_ST + simX[:, 5], label='Fz rear')
        # plt.title('Normal Force at the axis')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Nominal Force [N]')
        # # plt.ylim(1.1*-MAX_FX, 1.1*MAX_FX)
        # plt.legend()

        # plt.subplot(5,2,10)
        # plt.plot(time, simX[:, 5]*l/(m*h), label='ax')
        # plt.title('Longitudinal acceleration')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Longitudinal acceleration [m/s^2]')
        # # plt.ylim(-1.1*-MAX_FX, 1.1*MAX_FX)
        # plt.legend()

        plt.suptitle('MPC simulation results', fontsize=16)
        plt.tight_layout()
        plt.show(block=False)

    print(f'X0 = np.array([{", ".join(f"{x:.6f}" for x in X0)}])')


plt.show()