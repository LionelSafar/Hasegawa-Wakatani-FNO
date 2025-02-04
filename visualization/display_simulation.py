import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import jax.numpy as jnp

from visualization.posterior_plotting_tools import get_cmap

# Short script to visualize the data from the simulation

plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'lines.linewidth': 2,
    'grid.alpha': 0.5,
    'axes.prop_cycle': mpl.cycler(color=[
        '#1f77b4',
        '#ff7f0e', 
        '#9467bd',
        '#2ca02c',
        '#7f7f7f'])
})

datapath = '/scratch/izar/safar/example/sample_0_0.h5'
savepath = '/home/safar/'

with h5py.File(datapath, 'r') as f:
    # print all keys
    print("Keys: %s" % f.keys())
    phi = jnp.array(f['phi'][...])
    n = jnp.array(f['density'][...])
    omega = jnp.array(f['omega'][...])
    gamma_n = jnp.array(f['gamma_n'][...])
    gamma_c = jnp.array(f['gamma_c'][...])
    energy = jnp.array(f['energy'][...])
    enstrophy = jnp.array(f['enstrophy'][...])

time = jnp.arange(0, 8001) * 0.1
phisnap = phi[6000, :, :]
nsnap = n[6000, :, :]
L = 2 * jnp.pi / 0.15
box = [-L/2, L/2, -L/2, L/2]
cmap = get_cmap()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(time, enstrophy, color='tab:orange', label='Enstrophy')
ax[0, 0].plot(time, energy, color='tab:blue', label='Energy')

ax[0, 0].axvline(x=200, color='black', linestyle='--')
ax[0, 0].text(202, 0, 'Data truncation', fontsize=12)
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Energy/Enstrophy')
ax[0, 0].legend(loc='lower right')

ax[1, 0].plot(time, gamma_n, color='tab:green', label='Gamma_n')
ax[1, 0].plot(time, gamma_c, color='tab:red', label='Gamma_c')

ax[1, 0].axvline(x=200, color='black', linestyle='--')
ax[1, 0].text(202, 0, 'Data truncation', fontsize=12)
ax[1, 0].set_xlabel('Time')
ax[1, 0].set_ylabel(r'$\Gamma_n / \Gamma_c$')
ax[1, 0].legend(loc='lower right')

im1 = ax[0, 1].imshow(phisnap, cmap=cmap, extent=box)
ax[0, 1].set_title('$\phi$ snapshot')
ax[0, 1].grid(False)
im2 = ax[1, 1].imshow(nsnap, cmap=cmap, extent=box)
ax[1, 1].set_title('Density snapshot')
ax[1, 1].grid(False)

fig.colorbar(im1, ax=ax[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=ax[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

plt.tight_layout()

plt.savefig(savepath + 'DATA', format='pdf', dpi=300, bbox_inches='tight')


