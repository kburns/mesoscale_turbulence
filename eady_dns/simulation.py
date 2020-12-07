"""
DNS of the Eady problem, mostly following Basile's notes.
Linear and quadratic drag are both included.
"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
from parameters import *
from lowrank_fourier import rand_fourier_series_3d_lowrank

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

# Bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, L), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, L), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, H), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)
x, y, z = domain.all_grids()

# Problem
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'])
problem.parameters['L'] = L
problem.parameters['H'] = H
problem.parameters['κh'] = κh
problem.parameters['κv'] = κv
problem.parameters['νh'] = νh
problem.parameters['νv'] = νv
problem.parameters['f'] = f
problem.parameters['S'] = S
problem.parameters['N2'] = N2
problem.parameters['k1'] = k1
problem.parameters['k2'] = k2
problem.parameters['kp'] = kp
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
problem.substitutions['ωx'] = "wy - vz"
problem.substitutions['ωy'] = "uz - wx"
problem.substitutions['ωz'] = "vx - uy"
problem.substitutions['E'] = "(u*u + v*v + w*w) / 2"
problem.substitutions['Z'] = "(ωx*ωx + ωy*ωy + ωz*ωz) / 2"
problem.substitutions['left_mag_u'] = "sqrt(left(u)**2 + left(v)**2)"
problem.substitutions['ave(A)'] = "integ(A)/(L*L*H)"
problem.substitutions['Db'] = "κh*(dx(bx) + dy(by)) + κv*dz(bz)"
problem.substitutions['Du'] = "νh*(dx(ux) + dy(uy)) + νv*dz(uz)"
problem.substitutions['Dv'] = "νh*(dx(vx) + dy(vy)) + νv*dz(vz)"
problem.substitutions['Dw'] = "νh*(dx(wx) + dy(wy)) + νv*dz(wz)"
problem.add_equation("ux + vy + wz = 0")
problem.add_equation("dt(b) - Db + u0*bx + v*b0y + w*b0z    = - (u*bx + v*by + w*bz)")
problem.add_equation("dt(u) - Du + px - f*v + u0*ux + w*u0z = - (u*ux + v*uy + w*uz)")
problem.add_equation("dt(v) - Dv + py + f*u + u0*vx         = - (u*vx + v*vy + w*vz)")
problem.add_equation("dt(w) - Dw + pz - b   + u0*wx         = - (u*wx + v*wy + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(bz) = 0")
problem.add_bc("right(bz) = 0")
problem.add_bc("left(νv*uz - k1*u) = k2*left_mag_u*left(u)")
problem.add_bc("right(uz) = 0")
problem.add_bc("left(νv*vz - k1*v) = k2*left_mag_u*left(v)")
problem.add_bc("right(vz) = 0")
problem.add_bc("left(w - kp*ωz) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(w) = 0")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
logger.info('Solver built')

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    # Initial conditions
    b = solver.state['b']
    bz = solver.state['bz']
    rand = np.random.RandomState(978)
    b['g'] = init_amp * rand_fourier_series_3d_lowrank(x/L, y/L, z/H, kmax=init_kmax, density=1, rank=init_rank, rand=rand)
    b.differentiate('z', out=bz)
    # Timestepping and output
    initial_dt = max_dt
    fh_mode = 'overwrite'
else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)
    # Timestepping and output
    initial_dt = last_dt
    fh_mode = 'append'

# Analysis
checkpoints = solver.evaluator.add_file_handler('checkpoints', wall_dt=checkpoints_wall_dt, max_writes=1, mode=fh_mode)
checkpoints.add_system(solver.state)
slices = solver.evaluator.add_file_handler('slices', sim_dt=slices_sim_dt, max_writes=10, mode=fh_mode)
for field in ['b', 'u', 'v', 'w', 'ωz']:
    for loc in ['x=0', 'y=0', "z='left'", "z='center'", "z='right'"]:
        slices.add_task(f"interp({field}, {loc})")
scalars = solver.evaluator.add_file_handler('scalars', sim_dt=scalars_sim_dt, max_writes=100, mode=fh_mode)
scalars.add_task("ave(E)", name="<E>")
scalars.add_task("ave(Z)", name="<Z>")
scalars.add_task("ave(b*v)", name="<vb>")
scalars.add_task("ave(bz)", name="<bz>")
scalars.add_task("ave(b*Db)", name="<b*Db>")
scalars.add_task("ave(u*Du + v*Dv + w*Dw)", name="<u@Du>")
for n in [2,3,4]:
    scalars.add_task("ave(b**%i)" %n, name="<b^%i>" %n)
    scalars.add_task("ave(w**%i)" %n, name="<w^%i>" %n)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("ave(E)", name='<E>')

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
            logger.info('<E> = %.2e' %flow.max('<E>'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

