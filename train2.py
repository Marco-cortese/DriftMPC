import numpy as np
from numpy import pi as π, rad2deg as r2d, deg2rad as d2r
import matplotlib.pyplot as plt
from mpc import *
from numpy import linspace as linsp, logspace as logsp
from numpy.random import uniform as unf


DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples39.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples44.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples408.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples908.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples891.npz'
DS_FILE = 'data/dataset_stm_mpc_N100_Ts10ms_nsamples4516.npz'

ds = np.load(DS_FILE)
xs, us = ds['xs'], ds['us'] # shapes: (n_sample, n_horizon, 5), (n_sample, n_horizon, 2)
failed_x0s = ds['failed_x0s'] if 'failed_x0s' in ds else []
n_samples, n_horizon, nx = xs.shape
_, _, nu = us.shape

# normalize us
μ, σ = np.mean(us, axis=(0,1)), np.std(us, axis=(0,1))
us = (us - μ) / σ

print(f'Loaded dataset from {DS_FILE}, xs shape: {xs.shape}, us shape: {us.shape}')

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
    def __init__(self, input_dim, output_dim, hidden_dims=[32,32]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(Linear(prev_dim, hdim))
            layers.append(ActF(hdim))
            prev_dim = hdim
        layers.append(Linear(prev_dim, output_dim))
        self.net = Sequential(*layers)
    def forward(self, x): return self.net(x)

class DS(Dataset):
    def __init__(self, xs, us): self.xs, self.us = torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32)
    def __len__(self): return len(self.xs)*n_horizon
    def __getitem__(self, idx): 
        i1, i2 = divmod(idx, n_horizon)
        return self.xs[i1,i2,:], self.us[i1,i2,:]
    

## TRAINING
BS = 256 # batch size
EPOCHS = 100   # number of epochs
LR = 5e-2*logsp(0, -1, EPOCHS) 
MODEL_FILE = f'data/mlp_mpc.pt'

ds = DS(xs, us)
dl = DataLoader(ds, batch_size=BS, shuffle=True)

net = MLP(input_dim=nx, output_dim=nu, hidden_dims=[4,4])

opt = torch.optim.Adam(net.parameters(), lr=LR[0])
loss_fn = torch.nn.MSELoss()

print(f'Training... BS: {BS}, LR: {LR[::10]}, EPOCHS: {EPOCHS}')
best_loss = np.inf
for ep in range(EPOCHS):
    opt.param_groups[0]['lr'] = LR[ep]
    epoch_loss = 0.0
    for xb, ub in dl:
        pub = net(xb)
        loss = loss_fn(pub, ub)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(ds)
    if (ep+1)%5==0: print(f'Epoch {ep+1}/{EPOCHS}, Loss: {epoch_loss:.6f}')
    if epoch_loss < best_loss: best_loss = epoch_loss; torch.save(net.state_dict(), MODEL_FILE)
print(f'Training completed. Best loss: {best_loss:.6f}, model saved to {MODEL_FILE}')

#load model
net.load_state_dict(torch.load(MODEL_FILE))

## TESTING


# SIMULATION

mpc_t_ratio = 10  # MPC timestep is 10x the simulation timestep
ts_sim, T_sim = 0.001, 10 # simulation timestep and total simulation time
ts_mpc = mpc_t_ratio * ts_sim
# sim_model = STM_model_dt_inputs()
sim_model = DTM_model_dt_inputs()
sim = Simulator(sim_model=sim_model, ts_sim=ts_sim, integrator_type='ERK')
n_sim = int(T_sim / ts_sim) # simulation steps
assert (n_sim*ts_sim-T_sim) < 1e-9, "T_sim must be multiple of ts_sim"

v_lims = (2.0, 8.0)  # m/s # ficed limits
beta_lims = (-40*π/180, 40*π/180)
r_lims = (0, 60*π/180)

x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])  # initial condition
# x0 = np.array([4.5, d2r(-25), d2r(85), d2r(-13), 49.5])  # initial condition (close to target)
# x0 =  np.array([unf(*v_lims), unf(*beta_lims), unf(*r_lims), 0, 0])

simX = np.zeros((n_sim+1, nx))
simU = np.zeros((n_sim, nu))
simX[0,:] = x0
print(f'Simulating... n_sim: {n_sim}, ts_sim: {ts_sim}, ts_mpc: {ts_mpc}, x0: {x0}')
for i in range(n_sim):
    if i % mpc_t_ratio == 0:
        x_torch = torch.tensor(simX[i,:], dtype=torch.float32).view(1,-1)
        with torch.no_grad(): u_net = net(x_torch).view(nu)
        u_net = u_net.numpy()
        u_net = u_net * σ + μ  # denormalize
        du = u_net[:2]  # get (d_delta, d_Fx)
        du[0], du[1] = np.clip(du[0], -MAX_D_DELTA, MAX_D_DELTA), np.clip(du[1], -MAX_D_FX, MAX_D_FX)
    simU[i,:] = du
    next_x = sim.step(simX[i,:], du)
    #saturate delta and F_x
    next_x[3] = np.clip(next_x[3], -MAX_DELTA, MAX_DELTA)
    next_x[4] = np.clip(next_x[4], -MAX_FX, MAX_FX)
    simX[i+1,:] = next_x


# print(f'simX: {simX.shape}, simU: {simU.shape}')

# plot
t1 = np.arange(n_sim+1)*ts_sim
t2 = np.arange(n_sim)*ts_sim
plt.figure(figsize=(12,8))
plt.subplot(3,2,1)
plt.plot(t1, simX[:,0], label='V (m/s)')
plt.ylabel('V (m/s)'); plt.title('V (ms)'); plt.xlabel('Time (s)')
plt.subplot(3,2,2)
plt.plot(t1, r2d(simX[:,1]), label='beta (deg)')
plt.ylabel('beta (deg)'); plt.title('beta (deg)')
plt.subplot(3,2,3)
plt.plot(t1, r2d(simX[:,2]), label='r (deg/s)')
plt.ylabel('r (deg/s)'); plt.title('r (deg/s)')
plt.subplot(3,2,4)

plt.subplot(3,2,5)
plt.plot(t1, r2d(simX[:,3]), label='delta (deg)')
plt.plot(t2, r2d(simU[:,0])*ts_mpc, label='d_delta (deg/s)')
plt.ylabel('Steering rate'); plt.legend()
plt.subplot(3,2,6)
plt.plot(t1, simX[:,4], label='F_x (N)')
plt.plot(t2, simU[:,1]*ts_mpc, label='d_Fx (N/s)')
plt.ylabel('Longitudinal force'); plt.legend()

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