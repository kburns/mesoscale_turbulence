"""
Plot planes from joint analysis files.

Usage:
    plot_spectra.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./spectra]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
plt.ioff()

import dedalus.public as de
from dedalus.extras import plot_tools
from parameters import *

# Domain and scratch field
x_basis = de.Fourier('x', Nx, interval=(0, L), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, L), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, H), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
field = domain.new_field()

def integrated_spectra(field, axis):
    """Integrate power over specified axis."""
    power = np.abs(field['c'])**2
    return np.sum(power, axis=axis)

def main(filename, start, count, output):
    plot_spectra(filename, start, count, output)

def plot_spectra(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    tasks = ['b', 'u', 'v', 'w']
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'spec_{:06}.png'.format(write)
    # Layout
    nrows = 3
    ncols = len(tasks)
    image = plot_tools.Box(4, 4)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
    scale = 2
    dpi = 100
    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for j, task in enumerate(tasks):
                dset = file['tasks'][task]
                # Load field in grid layout
                field.set_scales(1)
                field['g'] = dset[index]
                for i in range(3):
                    data = np.log10(integrated_spectra(field, axis=i).T)
                    image_axes = [0, 1, 2]
                    image_axes.pop(i)
                    # Build subfigure axes
                    axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                    plot_tools.plot_bot(field, image_axes, data=data, axes=axes, title=task, even_scale=False,cmap='viridis')
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

