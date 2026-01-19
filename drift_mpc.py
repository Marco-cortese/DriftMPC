from mpc import *
from numpy.random import uniform as unf

# Equilibrium point (found in PYTHON) [V, beta, r, delta, Fx]
# x_eq, u_eq = [3.610849747542315, -0.4363323129985824, 1.2036165825141052], [-0.18825862766328222, 27.47665205296075]
# x_eq, u_eq = [4.486209860862883, -0.4363323129985824, 1.4954032869542941], [-0.11596738898598893, 46.64426852037662]

# initial condition for [V, beta, r, delta, Fx]
# X0 = np.array([9, 0.0, 0.0, 0.0, 0.0]) 
# X0 = np.array([3, 0.0, 0.0, 0.0, 0.0]) 
# X0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0]) 

# random ic
v_lims = (2.0, 8.0)  # m/s # ficed limits
# v_lims = (YREF[0]-1, YREF[0]+1)  # m/s # around ref
beta_lims = (-40*π/180, 40*π/180)
r_lims = (-100*π/180, 100*π/180)
X0 = np.array([unf(*v_lims), unf(*beta_lims), unf(*r_lims), 0, 0])

# hard ic
# X0 = np.array([4.037190, -0.109424, 1.235876, 0.000000, 0.000000])
# X0 = np.array([4.289183, -0.393230, -0.469857, 0.000000, 0.000000])

print(f"Initial condition: V={X0[0]:.2f} m/s, beta={np.rad2deg(X0[1]):.2f} deg, r={np.rad2deg(X0[2]):.2f} deg/s, delta={np.rad2deg(X0[3]):.2f} deg, Fx={X0[4]:.2f} N")

# setup controller parameters
Ts = 0.01 # - controller sampling time [s]
N  = 100 #100 # - number of shooting time intervals e
T = N*Ts # - prediction horizon length [s]
T_tot = 2.5 #10.0 # total simulation time [s]

# - system model
model = STM_model_dt_inputs(); x0=X0
# model = DTM_model_dt_inputs(); x0=X0
# model = DTM_model_LT_dt_inputs(Ts); x0=np.concatenate([X0, [0.0]])

# - simulation model
ts_sim = 0.001 # simulation fundamental time step [s] <-
# ts_sim = 0.01 # simulation fundamental time step [s]
# sim_model = model; x0_sim = x0  # use the same model for simulation
# sim_model = STM_model_dt_inputs_sim(); x0_sim=X0
sim_model = DTM_model_dt_inputs_sim(); x0_sim=X0
# sim_model = DTM_model_LT_dt_inputs_sim(ts_sim); x0_sim=np.concatenate([X0, [0.0]])

# get state and control dimensions
nx, nu = model.x.rows(), model.u.rows()
nx_sim, nu_sim = sim_model.x.rows(),  sim_model.u.rows()

# define reference trajectories
V_ref, Tf    = piecewise_constant([4.5],[T_tot], Ts)
# V_ref, Tf    = piecewise_constant([5, 3, 5],[T_tot/3, T_tot/3, T_tot/3], Ts)
beta_ref, _  = piecewise_constant([np.deg2rad(-30)],[T_tot], Ts)
# beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30) ],[T_tot/2, T_tot/2], Ts)
# beta_ref, _  = piecewise_constant([np.deg2rad(-30),np.deg2rad(30), np.deg2rad(-30) ],[T_tot/3, T_tot/3, T_tot/3], Ts)
r_ref, _     = piecewise_constant([0],[T_tot], Ts)

# compute the number of steps for simulation
N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

print(f'N_steps sim: {N_steps}, N_steps controller: {N_steps_dt}, n_update: {n_update}')

## SIMULATION

# setup simulation of system dynamics
sim = Simulator(sim_model=sim_model, ts_sim=ts_sim, integrator_type='ERK')

# set up MPC
mpc_ctrl = MPC_Controller(model, N, T)

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

print("Average MPC computation time: %.4f ms" % (1e3*np.mean(cpt)))
print("Average MPC cost: %.4f " % (np.mean(costs)))

## PLOTS

time        = np.linspace(0, ts_sim * N_steps, N_steps + 1)
time_mpc    = np.linspace(0, Ts * (N_steps_dt-1), N_steps_dt+1)

sim_V_ref = piecewise_constant(V_ref[:-1], [Ts]*(len(V_ref)-1), ts_sim)[0]
sim_beta_ref = piecewise_constant(beta_ref[:-1], [Ts]*(len(beta_ref)-1), ts_sim)[0]
sim_r_ref = piecewise_constant(r_ref[:-1], [Ts]*(len(r_ref)-1), ts_sim)[0]
errors = np.column_stack((sim_V_ref, sim_beta_ref, sim_r_ref)) - simX[:,:3]

if True:
# if False:
    # plot the simulation results
    CM = 'jet' #'inferno'
    fig = plt.figure(figsize=(20, 12))
    plt.subplot(5,2,1) # Velocity plots
    plt.plot(time, simX[:, 0], label='V')
    plt.plot(time_mpc, V_ref, linestyle=':', label='V_ref')
    plt.ylim(1.1*-MAX_V, 1.1*MAX_V)
    plt.title('Total Velocity'), plt.xlabel('Time (s)'), plt.ylabel('Total Velocity (m/s)'), plt.legend()
    plt.subplot(5,2,2)
    plt.plot(time, errors[:,0], label='error')
    plt.plot(time, np.zeros_like(time), linestyle=':')
    plt.title('Error Velocity'), plt.xlabel('Time (s)'), plt.ylabel('Error Velocity (m/s)')

    plt.subplot(5,2,3) # Sideslip angle plots
    plt.plot(time, np.rad2deg(simX[:, 1]), label='beta')
    plt.plot(time_mpc, np.rad2deg(beta_ref), linestyle=':', label='beta_ref')
    plt.ylim(-60, 60)
    plt.title('Sideslip Angle'), plt.xlabel('Time (s)'), plt.ylabel('Sideslip angle (deg)'), plt.legend()
    plt.subplot(5,2,4)
    plt.plot(time, np.rad2deg(errors[:,1]), label='error')
    plt.plot(time, np.zeros_like(time), linestyle=':')
    plt.title('Error Sideslip Angle'), plt.xlabel('Time (s)'), plt.ylabel('Error Sideslip Angle (deg)')

    plt.subplot(5,2,5) # Yaw rate plots
    plt.plot(time, np.rad2deg(simX[:, 2]), label='r')
    plt.plot(time_mpc, np.rad2deg(r_ref), linestyle=':', label='r_ref')
    plt.ylim(np.rad2deg(-4),np.rad2deg(4))
    plt.title('Yaw rate'), plt.xlabel('Time (s)'), plt.ylabel('Yaw rate (rad/s)'), plt.legend()
    plt.subplot(5,2,6)
    plt.plot(time, np.rad2deg(errors[:,2]), label='error')
    plt.plot(time, np.zeros_like(time), linestyle=':')
    plt.title('Error Yaw Rate'), plt.xlabel('Time (s)'), plt.ylabel('Error Yaw Rate (deg)')

    plt.subplot(5,2,7) # Control inputs plots
    plt.plot(time, np.rad2deg(simX[:, 3]), label='delta')
    plt.ylim(np.rad2deg(1.1*-MAX_DELTA), np.rad2deg(1.1*MAX_DELTA))
    plt.axhline(y=np.rad2deg(MAX_DELTA), color='gray', alpha=0.6), plt.axhline(y=-np.rad2deg(MAX_DELTA), color='gray', alpha=0.6)
    plt.title('Steering angle (at the ground)'), plt.xlabel('Time (s)'), plt.ylabel('Steering angle (deg)'), plt.legend()
    plt.subplot(5,2,8)
    plt.plot(time, simX[:, 4], label='Fx')
    plt.ylim(1.1*-MAX_FX, 1.1*MAX_FX)
    plt.axhline(y=MAX_FX, color='gray', alpha=0.6), plt.axhline(y=-MAX_FX, color='gray', alpha=0.6)
    plt.title('Rear wheel longitudinal force'), plt.xlabel('Time (s)'), plt.ylabel('Longitudinal Force (N)'), plt.legend()

    plt.subplot(5,2,9) # plot costs and CPU time
    plt.plot(time_mpc[:-1], costs, label='cost')
    plt.ylim(0, 1.1*max(costs))
    plt.title('MPC cost'), plt.xlabel('Time (s)'), plt.ylabel('Cost'), plt.legend()
    plt.subplot(5,2,10)
    plt.plot(time_mpc[:-1], 1e3*cpt, label='CPU time')
    plt.ylim(0, 1.1*max(1e3*cpt))
    plt.title('MPC computation time'), plt.xlabel('Time (s)'), plt.ylabel('Computation time (ms)'), plt.legend()

    # # load transfer plots
    # plt.subplot(5,2,9)
    # plt.plot(time, Fz_Front_ST - simX[:, 5], label='Fz front')
    # plt.plot(time, Fz_Rear_ST + simX[:, 5], label='Fz rear')
    # plt.title('Normal Force at the axis')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Nominal Force (N)')
    # # plt.ylim(1.1*-MAX_FX, 1.1*MAX_FX)
    # plt.legend()

    # plt.subplot(5,2,10)
    # plt.plot(time, simX[:, 5]*l/(m*h), label='ax')
    # plt.title('Longitudinal acceleration')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Longitudinal acceleration (m/s^2)')
    # # plt.ylim(-1.1*-MAX_FX, 1.1*MAX_FX)
    # plt.legend()

    plt.suptitle('MPC simulation results', fontsize=16)
    plt.tight_layout()
    plt.show(block=False)


# animation
try:
    anim = car_anim(
        xs=simX[:,:3],  # use the state vector as input
        us=piecewise_constant(simX[:-1,3:5], [ts_sim]*(len(simX)-1), ts_sim)[0],
        ic=np.array([0, 0, π/2]),  # initial conditions (x, y, ψ) 
        dt=ts_sim,  # time step
        fps=60,  # frames per second
        speed=.5,  # speed factor for the animation 
        title='MPC simulation results',  # title of the animation
        in_notebook=False,
    )  # run the car animation with the STM results
except Exception as e:
    print(f"Animation could not be created: {e}")

plt.show()

print("Final state:")
print(f"V: {simX[-1,0]:.2f} m/s")
print(f"Beta: {np.rad2deg(simX[-1,1]):.2f} deg")
print(f"Yaw rate: {np.rad2deg(simX[-1,2]):.2f} deg/s")
print(f"Delta: {np.rad2deg(simX[-1,3]):.2f} deg")
print(f"Fx: {simX[-1,4]:.2f} N")

print(f"Initial condition: V={X0[0]:.2f} m/s, beta={np.rad2deg(X0[1]):.2f} deg, r={np.rad2deg(X0[2]):.2f} deg/s, delta={np.rad2deg(X0[3]):.2f} deg, Fx={X0[4]:.2f} N")
print(f'X0 = np.array([{", ".join(f"{x:.6f}" for x in X0)}])')
