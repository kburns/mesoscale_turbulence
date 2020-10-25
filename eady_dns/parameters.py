
# Control parameters
Nx, Ny, Nz = 512, 512, 32
mesh = (16, 12)
A = 1000 # Aspect ratio: L / H
An = 3e4 # Diffusion anisotrophy: κh / κv
Ro = 0.25 # Rossby number: S / f
Ek = 1e-5 # Ekman number: νv / f H^2
Pr = 1 # Prandtl number: νv / κv = νh / κh
Ri = 1000 / Ro**2 # Richardson number: N^2 / S^2
D1 = 0.0016 # Linear drag parameter: k1 / f H
D2 = 0 # Quadratic drag parameter: k2
timestepper = "RK222"
stop_sim_time = 50000
max_dt = 10
checkpoints_sim_dt = 10000
slices_sim_dt = 100

# Fixed parameters
H = 1 # Domain height
f = 1 # Coriolis parameter

# Derived parameters
L = A * H # Domain length
κv = Ek / Pr * f * H**2 # Thermal diffusivity (vertical)
κh = An * κv # Thermal diffusivity (horizontal)
νv = Pr * κv # Kinematic viscosity (vertical)
νh = Pr * κh # Kinematic viscosity (horizontal)
S = Ro * f # Background vertical shear
N2 = Ri * S**2 # Background stratification
k1 = D1 * f * H # Linear bottom drag
k2 = D2 # Quadratic bottom drag

