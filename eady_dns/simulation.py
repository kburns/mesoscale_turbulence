"""
DNS of the Eady problem, mostly following Basile's notes.
Linear and quadratic drag are both included.
"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Background state:
#   p0x = 0
#   p0y = - f u0(z)
#   p0z = b0(y,z)
#   p0 = - f u0(z) y + c(z)
#   b0(y,z) = - f u0'(z) y + c'(z)

# Restricting to linear shear and stratification:
#   u0(z) = S * z
#   b0(y,z) = - f S y + c'(z)
#   b0z(z) = c''(z) = N2

# Control parameters
Nx, Ny, Nz = 128, 128, 64
A = 1000 # Aspect ratio: L / H
An = 3e4 # Diffusion anisotrophy: κh / κv
Ro = 0.25 # Rossby number: S / f
Ek = 1e-5 # Ekman number: νv / f H^2
Pr = 1 # Prandtl number: νv / κv = νh / κh
Ri = 1000 / Ro**2 # Richardson number: N^2 / S^2
D1 = 0.0016 # Linear drag parameter: k1 / f H
D2 = 0 # Quadratic drag parameter: k2
timestepper = "RK222"
stop_sim_time = 1000
max_dt = 10

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

# Bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, L), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, L), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, H), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
x, y, z = domain.all_grids()

# Problem
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'])
problem.parameters['κh'] = κh
problem.parameters['κv'] = κv
problem.parameters['νh'] = νh
problem.parameters['νv'] = νv
problem.parameters['f'] = f
problem.parameters['S'] = S
problem.parameters['N2'] = N2
problem.parameters['k1'] = k1
problem.parameters['k2'] = k2
problem.substitutions['u0'] = "S * z"
problem.substitutions['u0z'] = "S"
problem.substitutions['b0y'] = "- f * S"
problem.substitutions['b0z'] = "N2"
problem.substitutions['px'] = "dx(p)"
problem.substitutions['py'] = "dy(p)"
problem.substitutions['pz'] = "dz(p)"
problem.substitutions['bx'] = "dx(b)"
problem.substitutions['by'] = "dy(b)"
problem.substitutions['ux'] = "dx(u)"
problem.substitutions['uy'] = "dy(u)"
problem.substitutions['vx'] = "dx(v)"
problem.substitutions['vy'] = "dy(v)"
problem.substitutions['wx'] = "dx(w)"
problem.substitutions['wy'] = "dy(w)"
problem.substitutions['ωz'] = "vx - uy"
problem.add_equation("ux + vy + wz = 0")
problem.add_equation("dt(b) - κh*(dx(bx) + dy(by)) - κv*dz(bz) + u0*bx + v*b0y + w*b0z    = - (u*bx + v*by + w*bz)")
problem.add_equation("dt(u) - νh*(dx(ux) + dy(uy)) - νv*dz(uz) + px - f*v + u0*ux + w*u0z = - (u*ux + v*uy + w*uz)")
problem.add_equation("dt(v) - νh*(dx(vx) + dy(vy)) - νv*dz(vz) + py + f*u + u0*vx         = - (u*vx + v*vy + w*vz)")
problem.add_equation("dt(w) - νh*(dx(wx) + dy(wy)) - νv*dz(wz) + pz - b   + u0*wx         = - (u*wx + v*wy + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(bz) = 0")
problem.add_bc("right(bz) = 0")
problem.add_bc("left(νv*uz - k1*u) = left(k2*u**2)")
problem.add_bc("right(uz) = 0")
problem.add_bc("left(νv*vz - k1*v) = left(k2*v**2)")
problem.add_bc("right(vz) = 0")
problem.add_bc("left(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(w) = 0")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
logger.info('Solver built')

# Initial conditions
b = solver.state['b']
bz = solver.state['bz']
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]
# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
b['g'] = 1e-3 * noise * (zt - z) * (z - zb)
b.differentiate('z', out=bz)

# Analysis
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=100, max_writes=1)
checkpoints.add_system(solver.state)
slices = solver.evaluator.add_file_handler('slices', sim_dt=10, max_writes=10)
for field in ['b', 'u', 'v', 'w', 'ωz']:
    for loc in ['x=0', 'y=0', "z='left'", "z='center'", "z='right'"]:
        slices.add_task(f"interp({field}, {loc})")

# CFL
CFL = flow_tools.CFL(solver, initial_dt=max_dt, cadence=5, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=max_dt)
CFL.add_velocities(('u', 'v', 'w'))

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

