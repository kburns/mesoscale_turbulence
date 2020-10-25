
# Control parameters
Nx, Ny, Nz = 512, 512, 32
mesh = (16, 12)
A = 1000 # Aspect ratio: L / H
Ro = 0.25 # Rossby number: S / f
Ri = 1000 / Ro**2 # Richardson number: N^2 / S^2 = N^2 / f^2 Ro^2
Ek_v = 1e-5 # Ekman number (vertical): νv / f H^2
Ek_h = 3e-1 # Ekman number (horizontal): νh / f H^2
Pr = 1 # Prandtl number: νv / κv = νh / κh
D1 = 0.0016 # Linear drag parameter: k1 / f H
D2 = 0 # Quadratic drag parameter: k2
Ek_p = 0 # Ekman number (pumping closure): νp / f H^2 = 2 kp^2 / H^2
timestepper = "RK222"
safety = 0.5
stop_sim_time = 50000
max_dt = 10
checkpoints_wall_dt = 59*60
slices_sim_dt = 100

# Fixed parameters
H = 1 # Domain height
f = 1 # Coriolis parameter

# Derived parameters
L = A * H # Domain length
S = Ro * f # Background vertical shear
N2 = Ri * S**2 # Background stratification
νv = Ek_v * f * H**2 # Kinematic viscosity (vertical)
νh = Ek_h * f * H**2 # Kinematic viscosity (horizontal)
κv = νv / Pr # Thermal diffusivity (vertical)
κh = νh / Pr # Thermal diffusivity (horizontal)
k1 = D1 * f * H # Linear bottom drag
k2 = D2 # Quadratic bottom drag
kp = (Ek_p / 2)**0.5 * H # Ekman pumping closure scale

