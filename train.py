import numpy as np
from numpy import pi as π, rad2deg as r2d, deg2rad as d2r
import matplotlib.pyplot as plt

DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples39.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples44.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples408.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples908.npz'

ds = np.load(DS_FILE)
xs, us = ds['xs'], ds['us'] # shapes: (n_sample, n_horizon, 5), (n_sample, n_horizon, 2)
failed_x0s = ds['failed_x0s'] if 'failed_x0s' in ds else []
n_samples, n_horizon, nx = xs.shape
_, _, nu = us.shape

# normalize us
μ, σ = np.mean(us, axis=(0,1)), np.std(us, axis=(0,1))
us = (us - μ) / σ

print(f'Loaded dataset from {DS_FILE}, xs shape: {xs.shape}, us shape: {us.shape}')

# # 3d scatter the trajectory points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs[:,0,0], r2d(xs[:,0,1]), xs[:,0,2], c='b', marker='x', s=3)
# ax.scatter(failed_x0s[:,0], r2d(failed_x0s[:,1]), failed_x0s[:,2], c='r', marker='x', s=10) 
# # plot the entire trajectories
# for i in range(n_samples): ax.plot(xs[i,:,0], r2d(xs[i,:,1]), xs[i,:,2], alpha=0.7)
# ax.set_ylabel('beta (deg)')
# ax.set_zlabel('r (rad/s)')
# plt.show()

# # plot trajectories in the 2d projections
# v_lims, beta_lims, r_lims = (2.0, 8.0), (-40*π/180, 40*π/180), (0, 60*π/180)
# v_rect, beta_rect, r_rect = [v_lims[0], v_lims[1], v_lims[1], v_lims[0], v_lims[0]], \
#                              [beta_lims[0], beta_lims[0], beta_lims[1], beta_lims[1], beta_lims[0]], \
#                              [r_lims[0], r_lims[0], r_lims[1], r_lims[1], r_lims[0]]
# c_x0, c_rect, s_x0, lw = 'g', 'gray', 20, 1
# c_fx0, s_fx0 = 'r', 20
# plt.figure(figsize=(22,12))
# plt.subplot(1,3,1) # V vs beta
# # for i in range(n_samples): plt.plot(xs[i,:,0], r2d(xs[i,:,1]), alpha=0.7)
# plt.scatter(xs[:,0,0], r2d(xs[:,0,1]), c=c_x0, marker='x', s=s_x0)
# plt.scatter(failed_x0s[:,0], r2d(failed_x0s[:,1]), c=c_fx0, marker='x', s=s_fx0) 
# plt.plot(v_rect, r2d(beta_rect), c=c_rect, linewidth=lw)
# plt.xlabel('V (m/s)'); plt.ylabel('beta (deg)'); plt.grid(); plt.title('Trajectories: V vs beta')
# plt.subplot(1,3,2) # V vs r
# # for i in range(n_samples): plt.plot(xs[i,:,0], xs[i,:,2], alpha=0.7)
# plt.scatter(xs[:,0,0], xs[:,0,2], c=c_x0, marker='x', s=s_x0)
# plt.scatter(failed_x0s[:,0], failed_x0s[:,2], c=c_fx0, marker='x', s=s_fx0) 
# plt.plot(v_rect, r_rect, c=c_rect, linewidth=lw)
# plt.xlabel('V (m/s)'); plt.ylabel('r (rad/s)'); plt.grid(); plt.title('Trajectories: V vs r')
# plt.subplot(1,3,3) # beta vs r
# # for i in range(n_samples): plt.plot(r2d(xs[i,:,1]), xs[i,:,2], alpha=0.7)
# plt.scatter(r2d(xs[:,0,1]), xs[:,0,2], c=c_x0, marker='x', s=s_x0)
# plt.scatter(r2d(failed_x0s[:,1]), failed_x0s[:,2], c=c_fx0, marker='x', s=s_fx0)
# plt.plot(r2d(beta_rect), r_rect[::-1], c=c_rect, linewidth=lw)
# plt.xlabel('beta (deg)'); plt.ylabel('r (rad/s)'); plt.grid(); plt.title('Trajectories: beta vs r')
# plt.tight_layout()
# plt.show()
# raise

import torch
from torch.nn import functional as F, Module, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d, Tanh
from torch.utils.data import Dataset, DataLoader

class ActF(Module): # swish
    def __init__(self, no=1): 
        super(ActF, self).__init__()
        if no == 1: self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else: self.beta = torch.nn.Parameter(torch.ones(no), requires_grad=True) # n is the number of outputs
    def forward(self, x): return x*torch.sigmoid(self.beta*x)

class MLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32,32], actf=ActF):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(Linear(prev_dim, hdim))
            layers.append(actf())
            prev_dim = hdim
        layers.append(Linear(prev_dim, output_dim))
        self.net = Sequential(*layers)
    def forward(self, x): return self.net(x)

class DS(Dataset):
    def __init__(self, xs, us): self.xs, self.us = torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32)
    def __len__(self): return len(self.xs)
    def __getitem__(self, idx): return self.xs[idx,0,:3], self.us[idx]
    

## TRAINING
BS = 32 # batch size
LR = 3e-2 # learning rate
NEP = 500   # number of epochs
MODEL_FILE = f'data/mlp_mpc.pt'


ds = DS(xs, us)
dl = DataLoader(ds, batch_size=BS, shuffle=True)

net = MLP(input_dim=3, output_dim=nu*n_horizon, hidden_dims=[16,16], actf=ActF)

opt = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

print(f'Training... BS: {BS}, LR: {LR}, NEP: {NEP}')
for ep in range(NEP):
    epoch_loss = 0.0
    for xb, ub in dl:
        pub = net(xb).view(-1, n_horizon, nu)
        loss = loss_fn(pub, ub)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(ds)
    if (ep+1)%10==0: print(f'Epoch {ep+1}/{NEP}, Loss: {epoch_loss:.6f}')
# save model
torch.save(net.state_dict(), MODEL_FILE)

#load model
net.load_state_dict(torch.load(MODEL_FILE))

## TESTING
nraw, ncol = 8, 5
n_test = nraw//2 * ncol
test_xs = xs[:n_test,:, :3]
test_us = us[:n_test,:,:]
with torch.no_grad():
    test_x_torch = torch.tensor(test_xs[:,0,:], dtype=torch.float32)
    pred_us_torch = net(test_x_torch).view(-1, n_horizon, nu)
    pred_us = pred_us_torch.numpy()
plt.figure(figsize=(32,18))
for i in range(n_test):
    plt.subplot(nraw, ncol, i+1)
    plt.plot(pred_us[i,:,0], label='Predicted d_delta')
    plt.plot(test_us[i,:,0], label='True d_delta', linestyle='dashed')
    plt.legend()
    plt.subplot(nraw, ncol, i+1+n_test)
    plt.plot(pred_us[i,:,1], label='Predicted d_Fx')
    plt.plot(test_us[i,:,1], label='True d_Fx', linestyle='dashed')
    plt.legend()
plt.tight_layout()



# # SIMULATION

# from mpc import Simulator, STM_model_dt_inputs, piecewise_constant, car_anim
# mpc_t_ratio = 10  # MPC timestep is 10x the simulation timestep
# ts_sim, T_sim = 0.001, 2.5 # simulation timestep and total simulation time
# ts_mpc = mpc_t_ratio * ts_sim
# sim = Simulator(sim_model=STM_model_dt_inputs(), ts_sim=ts_sim, integrator_type='ERK')
# n_sim = int(T_sim / ts_sim) # simulation steps
# assert (n_sim*ts_sim-T_sim) < 1e-9, "T_sim must be multiple of ts_sim"
# x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])  # initial condition
# simX = np.zeros((n_sim+1, nx))
# simU = np.zeros((n_sim, nu))
# simX[0,:] = x0
# print(f'Simulating... n_sim: {n_sim}, ts_sim: {ts_sim}, ts_mpc: {ts_mpc}, x0: {x0}')
# for i in range(n_sim):
#     if i % mpc_t_ratio == 0:
#         x_torch = torch.tensor(simX[i,:3], dtype=torch.float32).view(1,-1)
#         with torch.no_grad(): u_torch = net(x_torch).view(n_horizon, nu)
#         u = u_torch[0,:].numpy()
#     simU[i,:] = u
#     simX[i+1,:] = sim.step(simX[i,:], u)

# print(f'simX: {simX.shape}, simU: {simU.shape}')

# # plot
# t1 = np.arange(n_sim+1)*ts_sim
# t2 = np.arange(n_sim)*ts_sim
# plt.figure(figsize=(12,8))
# plt.subplot(3,2,1)
# plt.plot(t1, simX[:,0], label='V (m/s)')
# plt.ylabel('V (m/s)'); plt.title('V (ms)'); plt.xlabel('Time (s)')
# plt.subplot(3,2,2)
# plt.plot(t1, r2d(simX[:,1]), label='beta (deg)')
# plt.ylabel('beta (deg)'); plt.title('beta (deg)')
# plt.subplot(3,2,3)
# plt.plot(t1, r2d(simX[:,2]), label='r (deg/s)')
# plt.ylabel('r (deg/s)'); plt.title('r (deg/s)')
# plt.subplot(3,2,4)

# plt.subplot(3,2,5)
# plt.plot(t1, r2d(simX[:,3]), label='delta (deg)')
# plt.plot(t2, r2d(simU[:,0])*ts_mpc, label='d_delta (deg/s)')
# plt.ylabel('Steering rate'); plt.legend()
# plt.subplot(3,2,6)
# plt.plot(t1, simX[:,4], label='F_x (N)')
# plt.plot(t2, simU[:,1]*ts_mpc, label='d_Fx (N/s)')
# plt.ylabel('Longitudinal force'); plt.legend()

# plt.tight_layout()
# plt.show(block=False)

# # animation
# try:
#     anim = car_anim(
#         xs=simX[:,:3],  # use the state vector as input
#         us=piecewise_constant(simX[:-1,3:5], [ts_sim]*(len(simX)-1), ts_sim)[0],
#         ic=np.array([0, 0, π/2]),  # initial conditions (x, y, ψ) 
#         dt=ts_sim,  # time step
#         fps=60,  # frames per second
#         speed=.5,  # speed factor for the animation 
#         title='MPC simulation results',  # title of the animation
#         in_notebook=False,
#     )  # run the car animation with the STM results
# except Exception as e:
#     print(f"Animation could not be created: {e}")

plt.show()