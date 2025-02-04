import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from utils.trainstate_init import ExtendedTrainState
from utils.physical_quantities import (
    get_physics,
    periodic_gradient,
    periodic_laplace,
    HW_residue,
)
from utils.data_handling import split_into_shorter_sequences, seq_to_X


# This file contains all functions used in model_evaluation.py to plot the results of the models
# in post-training comparison of multiple models


def get_cmap()->LinearSegmentedColormap:
    """
    returns modified icefire colormap from Seaborn for spatial scale
    spectral for frequency scale

    """
    cmap = sns.color_palette("icefire", as_cmap=True)
    colors = cmap(np.linspace(0, 1, 256))
    colors = np.clip(colors * 1.4, 0, 1)
    cmap = LinearSegmentedColormap.from_list("modified_icefire", colors)

    return cmap


def get_rel_l2_loss(y_true: jnp.ndarray, y_pred:jnp.ndarray) -> float:
    """
    Get the relative L2 loss between the true and predicted sequences.
    
    Args:
        y_true (jnp.ndarray): True sequence
        y_pred (jnp.ndarray): Predicted sequence

    """
    return jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

def plot_snaps(
        seq_dict: dict, 
        param_dict: dict,
        variable: str, 
        dx: float,
        in_images: int, 
        inc_error: bool = False,
        foldername: str = None, 
        mode: str = 'basic', 
        id: str = None
) -> None:
    """
    Plot snapshots of the predicted and true sequences at some snapshots. Based on the mode,
    the sequence is transformed before plotting.
    
    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences
        --> structure {'model': {'pred': pred, 'DNS': DNS}}
        param_dict (Dict): Dictionary containing the parameters of the model
        --> structure {'model': {'dt': dt, 'dx': dx}}
        variable (str): Variable to plot
        --> options 'phi', 'n'
        dx (float): spatial resolution (dx = 2*jnp.pi / resolution / 0.15)
        in_images (int): Number of input images to the model -> used to set the time axis
        inc_error (bool): Whether to include the error between the predicted and true sequence
        foldername (str): Folder to save the plots
        mode (str): Mode to plot the sequence
        --> options 'basic', 'fft_magnitude', 'fft_phase', 'fft_real', 'fft_imag', 'gradient', 'laplace'
        id (str): Identifier for the plot name

    """
    # Create a colormap from the list of colors
    seq_dict = seq_dict.copy()
    cmap_fft = cm.get_cmap('Spectral', 8)
    cmap = get_cmap()

    L = 2*jnp.pi / 0.15 # length of the domain
    resolution = L / dx
    variables = {'phi': 0, 'n': 1}
    k = variables[variable]
    M = seq_dict['DNS']['DNS'].shape[0] # number of timesteps

    # Only keep the chosen variable in each sequence
    seq_dict = {
            outer_key: {inner_key: value[..., k] for inner_key, value in outer_value.items()}
            for outer_key, outer_value in seq_dict.items()}

    # Depending on mode, transform the sequence
    if mode == 'basic':
        cmap = cmap
        add = ''
        box = [-L/2, L/2, -L/2, L/2]
    elif mode == 'fft_magnitude':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_magnitude(seq_dict[key]['pred'], axes=(-2, -1))
            seq_dict[key]['DNS'] = fft_magnitude(seq_dict[key]['DNS'], axes=(-2, -1))
            seq_dict[key]['pred'] = seq_dict[key]['pred'][:, 32:-32, 32:-32] # remove 1/4 of all sides
            seq_dict[key]['DNS'] = seq_dict[key]['DNS'][:, 32:-32, 32:-32]
            cmap=cmap_fft
            add = 'fft_magn'
            box = [-resolution * jnp.pi / L, resolution * jnp.pi / L, -resolution * jnp.pi / L, resolution * jnp.pi / L]
    elif mode == 'fft_phase':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_phase(seq_dict[key]['pred'], axes=(-2, -1))
            seq_dict[key]['DNS'] = fft_phase(seq_dict[key]['DNS'], axes=(-2, -1))
            cmap = cmap_fft
            add = 'fft_phase'
            box = [-resolution * jnp.pi / L, resolution * jnp.pi / L, -resolution * jnp.pi / L, resolution * jnp.pi / L]
    elif mode == 'fft_real':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_real(seq_dict[key]['pred'], axes=(-2, -1))
            seq_dict[key]['DNS'] = fft_real(seq_dict[key]['DNS'], axes=(-2, -1))
            cmap = cmap_fft
            add = 'fft_real'
            box = [-resolution * jnp.pi / L, resolution * jnp.pi / L, -resolution * jnp.pi / L, resolution * jnp.pi / L]
    elif mode == 'fft_imag':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_imag(seq_dict[key]['pred'], axes=(-2, -1))
            seq_dict[key]['DNS'] = fft_imag(seq_dict[key]['DNS'], axes=(-2, -1))
            cmap = cmap_fft
            add = 'fft_imag'
            box = [-resolution * jnp.pi / L, resolution * jnp.pi / L, -resolution * jnp.pi / L, resolution * jnp.pi / L]
    elif mode == 'gradient':
        for key in seq_dict.keys():
            x_grad = periodic_gradient(seq_dict[key]['pred'], dx, axis=-1)
            y_grad = periodic_gradient(seq_dict[key]['pred'], dx, axis=-2)
            seq_dict[key]['pred'] = x_grad + y_grad
            cmap = cmap
            add = 'grad'
            box = [-L/2, L/2, -L/2, L/2]
    elif mode == 'laplace':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = periodic_laplace(seq_dict[key]['pred'], dx)
            cmap = cmap
            add = 'laplacian'
            box = [-L/2, L/2, -L/2, L/2]


    N = len(seq_dict)
    K = 0
    if inc_error:
        K = len(seq_dict) - 1

    snaps = [in_images, in_images+9, in_images+19, 79, M-1] # adjust for timesnap
    ims = [] # store all images for colorbar
    fig, axs = plt.subplots(N+K, len(snaps), figsize=(15, 15), 
                            sharex=True, sharey=True, constrained_layout=True)
    #vmin = jnp.min(seq_dict['DNS']['DNS'][snaps, :, :]) 
    #vmax = jnp.max(seq_dict['DNS']['DNS'][snaps, :, :])
    for i, key in enumerate(seq_dict.keys()):
        dt = param_dict[key]['dt']
        times = jnp.round(jnp.arange(0, M*dt, dt), 2)
        times = times - (in_images-1)*dt
        for j, snap in enumerate(snaps):
            seq = seq_dict[key]['pred'][snap, :, :]
            if j == 0:
                axs[i, j].set_ylabel(fr"$\bf{{{key}}}$", fontsize=22)
            im = axs[i, j].imshow(seq, cmap=cmap, extent=box)
            # colorbar to all plot -> just for now..
            #plt.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
            ims.append(im)
            axs[i, j].grid(False)
            axs[i, j].tick_params(axis='x', labelsize=15) 
            axs[i, j].tick_params(axis='y', labelsize=15)  

            if i==0: # add time information to top row
                axs[i, j].set_title(f"t = {times[snap]:.2f} - {snap-in_images+1}$\\Delta t_{{NN}}$", fontsize=18)
        cbar_ax = fig.add_axes([0.0385, -0.02, 0.959, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=16)  # Adjust tick font size

    if inc_error: # Add rows for the error of each model
        for i, key in enumerate(seq_dict.keys()):
            if i == 0: # skip DNS
                continue
            errors = jnp.abs(seq_dict[key]['DNS'] - seq_dict[key]['pred'])
            for j, snap in enumerate(snaps):
                seq = errors[snap, :, :]
                if j == 0:
                    axs[N-1+i, j].set_ylabel(f'Error {key}', fontsize=14)
                im = axs[N-1+i, j].imshow(seq, cmap=cmap, extent=box)
                plt.colorbar(im, ax=axs[N-1+i, j], fraction=0.046, pad=0.04)

    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if foldername:
        if id:
            filename = f'{variable}_snaps_{add}_{id}.pdf'
        else:
            filename = f'{variable}_snaps_{add}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

# Transformation functions for the sequences
def fft_magnitude(sequence: jnp.ndarray, axes: Tuple=(-2, -1)) -> jnp.ndarray:
    """ Compute the magnitude of the 2D FFT of the sequence."""
    return jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(sequence, axes=axes), axes=axes))

def fft_phase(sequence: jnp.ndarray, axes: Tuple=(-2, -1)) -> jnp.ndarray:
    """ Compute the phase of the 2D FFT of the sequence."""
    return jnp.angle(jnp.fft.fftshift(jnp.fft.fft2(sequence, axes=axes), axes=axes))

def fft_real(sequence: jnp.ndarray, axes: Tuple=(-2, -1)) -> jnp.ndarray:
    """ Compute the real part of the 2D FFT of the sequence."""
    return jnp.real(jnp.fft.fftshift(jnp.fft.fft2(sequence, axes=axes), axes=axes))

def fft_imag(sequence:jnp.ndarray, axes: Tuple=(-2, -1)) -> jnp.ndarray:
    """ Compute the imaginary part of the 2D FFT of the sequence."""
    return jnp.imag(jnp.fft.fftshift(jnp.fft.fft2(sequence, axes=axes), axes=axes))


def plot_energy_cascade(
        seq_dict: Dict, 
        dx: float, 
        timelength: int=None, 
        foldername: str=None, 
        id: str=None
) -> None:
    """
    Plot the energy spectrum of the predicted and true data.
    Additionally plot the energy spectrum for each model in case of different regimes.
    
    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences
        dx (float): spatial resolution
        timelength (int): Number of timesteps to consider
        foldername (str): Folder to save the plots
        id (str): Identifier for the plot name

    """
    if timelength is None:
        timelength = seq_dict['DNS']['DNS'].shape[0] # If no snapshot is given, take the last one

    seq_dict = seq_dict.copy()

    # Set up line styles for the different models
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    # get wavenumber magnitude
    length = seq_dict['DNS']['DNS'].shape[1]
    kx = jnp.fft.fftfreq(length, d=dx) 
    ky = jnp.fft.fftfreq(length, d=dx)
    kx *= 2*jnp.pi # rescale to physical units
    ky *= 2*jnp.pi
    kx, ky = jnp.meshgrid(kx, ky, indexing='ij')
    k = jnp.sqrt(kx**2 + ky**2)  # wavenumber magnitude

    # Create bins for the energy spectrum
    k_bins = jnp.logspace(jnp.log(0.2), jnp.log(k.max()), 150)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    # Plot the energy spectrum for each model
    _, ax = plt.subplots()

    k_data = {}
    Ek_data = {}
    Ek_true_data = {}
    for i, key in enumerate(seq_dict.keys()):
        # Get fields
        seq = seq_dict[key]['pred'][:, :, :, :]
        seq_true = seq_dict[key]['DNS'][:, :, :, :]
        
        phi = seq[:, :, :, 0]
        n = seq[:, :, :, 1]

        phi_true = seq_true[:, :, :, 0]
        n_true = seq_true[:, :, :, 1]

        # Get E(k)
        n_hat = jnp.fft.fft2(n, axes=(-2, -1))
        phi_hat = jnp.fft.fft2(phi, axes=(-2, -1))

        n_hat_true = jnp.fft.fft2(n_true, axes=(-2, -1))
        phi_hat_true = jnp.fft.fft2(phi_true, axes=(-2, -1))

        Ek = (jnp.abs(n_hat)**2 + k**2 * jnp.abs(phi_hat)**2 ) / 2
        Ek_true = (jnp.abs(n_hat_true)**2 + k**2 * jnp.abs(phi_hat_true)**2 ) / 2

        # map over timesteps to get average after
        Ek_binned = vmap(bin_function, in_axes=(None, 0, None))(k, Ek, k_bins)
        Ek_true_binned = vmap(bin_function, in_axes=(None, 0, None))(k, Ek_true, k_bins)

        # Remove empty bins
        Ek_binned = jnp.where(Ek_binned == 0, jnp.nan, Ek_binned)
        Ek_binned = jnp.nanmean(Ek_binned, axis=0)
        print('EK bin shape after averaging', Ek_binned.shape)
        Ek_true_binned = jnp.where(Ek_true_binned == 0, jnp.nan, Ek_true_binned)
        Ek_true_binned = jnp.nanmean(Ek_true_binned, axis=0)

        # Get all empyt bins and remove
        mask = Ek_binned > 0
        k_binned = k_bin_centers[mask]
        Ek_binned = Ek_binned[mask]

        Ek_true_binned = Ek_true_binned[mask]

        k_data[key] = k_binned
        Ek_data[key] = Ek_binned
        Ek_true_data[key] = Ek_true_binned

        ax.plot(k_binned, Ek_binned, dashes=line_styles[i], label=key)
    ax.set(xlabel=r"$k$", ylabel=r'$\hat{E}(k)$', xscale='log', yscale='log')
    ax.set_xlim(left=0.6)
    ax.set_ylim([1e2, 1e9])
    ax.legend(title=f"Model", title_fontsize='large')
    ax.grid(True)
    plt.tight_layout()

    if foldername:
        if id:
            filename = f'/energy_spectrum_{id}.pdf'
        else:
            filename = '/energy_spectrum.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    # Plot DNS with values for each regime for comparison in different phyiscal regimes
    _, ax = plt.subplots()
    set1 = cm.get_cmap("Set1")
    colors = [set1(i) for i in range(8)]
    for i, key in enumerate(Ek_data.keys()):
        if i == 0:
            continue
        Ek_binned = Ek_data[key]
        k_binned = k_data[key]
        Ek_binned_DNS = Ek_true_data[key]
        ax.plot(k_binned, Ek_binned_DNS, color=colors[i-1], linewidth=1, alpha=0.9, linestyle=':')
        ax.plot(k_binned, Ek_binned, color=colors[i-1], dashes=line_styles[i], label=key)

    ax.set(xlabel=r"$k$", ylabel=r'$\hat{E}(k)$', xscale='log', yscale='log')
    ax.set_xlim(left=0.6)
    ax.set_ylim([1e2, 1e9])
    ax.legend(title=f"Configuration", title_fontsize='large')
    plt.tight_layout()

    if foldername:
        if id:
            filename = f'/energy_spectrum_multiregime_{id}.pdf'
        else:
            filename = '/energy_spectrum_multiregime.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')

def bin_function(k: jnp.ndarray, fk: jnp.ndarray, k_bins: jnp.ndarray) -> jnp.ndarray:
    """
    Bin the energy spectrum for a given snapshot of k and E(k) to binned radial k.

    Args:
        k (jnp.ndarray): Wavenumber magnitude as 2D image
        fk (jnp.ndarray): Energy spectrum as 2D image
        k_bins (jnp.ndarray): Bin range array for the wavenumber magnitude

    Returns:
        f_k_binned (jnp.ndarray): Binned energy spectrum
    
    """
    fk_flat = fk.flatten()
    k_flat = k.flatten()

    digitized = jnp.digitize(k_flat, k_bins)  # Find the bin index for each k
    f_k_binned = jnp.array([fk_flat[digitized == i].sum() for i in range(len(k_bins))])

    return f_k_binned[:-1]  # Exclude the last empty bin

def plot_loss_with_time(
        seq_dict: dict, 
        param_dict: dict,
        dx: float, 
        in_images: int, 
        mode: str='basic', 
        foldername: str=None, 
        id: str=None
) -> None:
    """
    Plot the relative L2 loss of the predicted and true sequences over time to display error accumulation.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        dx (float): physical grid spacing
        in_images (int): Number of input images to the model -> used to set the time axis
        mode (str): Mode to plot the sequence
            --> options 'basic', 'gradient', 'laplace', 'fft_magnitude', 'fft_phase'
        foldername (str): Folder to save the plots
        id (str): Identifier for the plot name
    
    """
    seq_dict = seq_dict.copy()

    # Get time and rescale to set t=0 before the first prediction = in_images - 1
    seq_dict = {outer_key: {inner_key: value[in_images-1:, :, :, :] for inner_key, value in outer_value.items()}
                for outer_key, outer_value in seq_dict.items()}
    
    # Set up line styles for the different models
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    # set field "mode" that is plotted
    mode = mode.lower()
    if mode == 'basic':
        add = ''
    elif mode == 'gradient':
        for key in seq_dict.keys():
            phi = seq_dict[key]['pred'][:, :, :, 0]
            n = seq_dict[key]['pred'][:, :, :, 1]
            phi_grad = periodic_gradient(phi, dx, axis=-1) + periodic_gradient(phi, dx, axis=-2)
            n_grad = periodic_gradient(n, dx, axis=-2) + periodic_gradient(n, dx, axis=-1)
            seq = jnp.stack([phi_grad, n_grad], axis=-1)
            seq_dict[key]['pred'] = seq
            add = 'grad'
    elif mode == 'laplace':
        for key in seq_dict.keys():
            phi = seq_dict[key]['pred'][:, :, :, 0]
            n = seq_dict[key]['pred'][:, :, :, 1]
            phi_lap = periodic_laplace(phi, dx)
            n_lap = periodic_laplace(n, dx)
            seq = jnp.stack([phi_lap, n_lap], axis=-1)
            seq_dict[key]['pred'] = seq
            add = 'laplacian'
    elif mode == 'fft_magnitude':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_magnitude(seq_dict[key]['pred'], axes=(1, 2))
            seq_dict[key]['pred'] = seq_dict[key]['pred'][:, 32:-32, 32:-32, :]
            seq_dict[key]['DNS'] = seq_dict[key]['DNS'][:, 32:-32, 32:-32, :]
            add = 'fft_magn'
    elif mode == 'fft_phase':
        for key in seq_dict.keys():
            seq_dict[key]['pred'] = fft_phase(seq_dict[key]['pred'], axes=(1, 2))
            add = 'fft_phase'
    else:
        raise ValueError("Mode not recognised. Choose from 'basic', " 
                         "'gradient', 'laplace', 'fft_magnitude', 'fft_phase'")    

    # set colors
    cmap =['#1f77b4','#ff7f0e', '#9467bd','#2ca02c','#7f7f7f']

    # Plot the relative L2 loss over time for each model
    _, ax = plt.subplots()
    for i, key in enumerate(seq_dict.keys()):
        if key == 'DNS': # skip DNS
            continue

        # Get time settings for each
        dt = param_dict[key]['dt']
        N = seq_dict[key]['DNS'].shape[0] # time length
        times = jnp.linspace(0, N*dt, N)

        # Calculate the relative L2 loss over space for each timestep and plot
        seq = seq_dict[key]['pred'][:, :, :, :]
        DNS_seq = seq_dict[key]['DNS'][:, :, :, :]
        loss = jnp.linalg.norm(seq-DNS_seq, axis=(1, 2)) / jnp.linalg.norm(DNS_seq, axis=(1, 2))
        loss = jnp.mean(loss, axis=-1)
        ax.plot(times, loss, color=cmap[i], dashes=line_styles[i], label=key)
    ax.set(xlabel=r'time $t$', ylabel=r'relative $l_2$ loss')
    ax.grid(True)
    ax.set_xlim(0, 75)
    ax.legend(title='Model', title_fontsize='large')

    plt.tight_layout()
    if foldername:
        if id:
            filename = f'/time_losses_{add}_{id}.pdf'
        else:
            filename = f'/time_losses_{add}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    
def plot_physics(
        seq_dict: Dict, 
        param_dict: Dict,
        dx: float, 
        foldername: str=None
) -> None:
    """
    Plot energy, enstrophy and the fluxes as time-series for each model and get the relative errors.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        dx (float): spatial resolution
        foldername (str): Folder to save the plots
    
    """
    seq_dict = seq_dict.copy()
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    # Print header
    print(50*'-')
    print('PHYSICS:')
    print(50*'-')

    fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
    for i, key in enumerate(seq_dict.keys()):
        dt = param_dict[key]['dt']
        nu = param_dict[key]['nu']
        kappa = param_dict[key]['kappa']

        # Transpose sequence to match the format of get_physics
        seq = seq_dict[key]['pred'].transpose(1, 2, 0, 3)
        seq_gt = seq_dict[key]['DNS'].transpose(1, 2, 0, 3)
        gt, pred = get_physics(
                seq, seq_gt, dt, dx, 1 ,nu, kappa, hyperdiff=3, inc_vorticity=False
                )

        # Get rolling statistics + time
        energy_mean, energy_std = rolling_stats(pred['energy'], 11)
        enstrophy_mean, enstrophy_std = rolling_stats(pred['enstrophy'], 11)
        gamma_n_mean, gamma_n_std = rolling_stats(pred['gamma_n'], 11)
        gamma_c_mean, gamma_c_std = rolling_stats(pred['gamma_c'], 11)
        times = jnp.round(jnp.linspace(0, pred['energy'].shape[0]*dt, pred['energy'].shape[0]), 2)
        time = times[5:-5] # remove first and last 5 due to rolling statistics

        # get mean and std of the total time series of predicted and DNS data
        E_mean_tot = jnp.mean(pred['energy'])
        E_std_tot = jnp.std(pred['energy'])
        U_mean_tot = jnp.mean(pred['enstrophy'])
        U_std_tot = jnp.std(pred['enstrophy'])
        G_n_mean_tot = jnp.mean(pred['gamma_n'])
        G_n_std_tot = jnp.std(pred['gamma_n'])
        G_c_mean_tot = jnp.mean(pred['gamma_c'])
        G_c_std_tot = jnp.std(pred['gamma_c']) 

        E_mean_tot_gt = jnp.mean(gt['energy'])
        E_std_tot_gt = jnp.std(gt['energy'])
        U_mean_tot_gt = jnp.mean(gt['enstrophy'])
        U_std_tot_gt = jnp.std(gt['enstrophy'])
        G_n_mean_tot_gt = jnp.mean(gt['gamma_n'])
        G_n_std_tot_gt = jnp.std(gt['gamma_n'])
        G_c_mean_tot_gt = jnp.mean(gt['gamma_c'])
        G_c_std_tot_gt = jnp.std(gt['gamma_c'])

        # Print statistics
        print(50*'-')
        print('MODEL:', key)
        print('E:', E_mean_tot, E_std_tot)
        print('U:', U_mean_tot, U_std_tot)
        print('G_n:', G_n_mean_tot, G_n_std_tot)
        print('G_c:', G_c_mean_tot, G_c_std_tot)
        print(50*'-')
        print('DNS:')
        print('E:', E_mean_tot_gt, E_std_tot_gt)
        print('U:', U_mean_tot_gt, U_std_tot_gt)
        print('G_n:', G_n_mean_tot_gt, G_n_std_tot_gt)
        print('G_c:', G_c_mean_tot_gt, G_c_std_tot_gt)
        print('RELATIVE ERRORS:')
        print('E:', jnp.abs((E_mean_tot - E_mean_tot_gt) / E_mean_tot_gt))
        print('U:', jnp.abs((U_mean_tot - U_mean_tot_gt) / U_mean_tot_gt))
        print('G_n:', jnp.abs((G_n_mean_tot - G_n_mean_tot_gt) / G_n_mean_tot_gt))
        print('G_c:', jnp.abs((G_c_mean_tot - G_c_mean_tot_gt) / G_c_mean_tot_gt))
        print(50*'-')

        # Plot the time series and display errors 
        axs[0, 0].plot(time, energy_mean, dashes=line_styles[i], label=f'$\mathbf{{{key}}}:$\n'
                       +f'$-E: {E_mean_tot:.2f} \pm {E_std_tot:.2f}$\n'
                       +f'$-U: {U_mean_tot:.2f} \pm {U_std_tot:.2f}$\n'
                       +f'$-\Gamma_n: {G_n_mean_tot:.2f} \pm {G_n_std_tot:.2f}$\n'
                       +f'$-\Gamma_c: {G_c_mean_tot:.2f} \pm {G_c_std_tot:.2f}$')

        axs[0, 0].fill_between(time, energy_mean - energy_std, energy_mean + energy_std, alpha=0.3)

        axs[0, 1].plot(time, enstrophy_mean, dashes=line_styles[i], label=key)
        axs[0, 1].fill_between(time, enstrophy_mean - enstrophy_std, enstrophy_mean + enstrophy_std, alpha=0.3)

        axs[1, 0].plot(time, gamma_n_mean, dashes=line_styles[i], label=key)
        axs[1, 0].fill_between(time, gamma_n_mean - gamma_n_std, gamma_n_mean + gamma_n_std, alpha=0.3)

        axs[1, 1].plot(time, gamma_c_mean, dashes=line_styles[i], label=key)
        axs[1, 1].fill_between(time, gamma_c_mean - gamma_c_std, gamma_c_mean + gamma_c_std, alpha=0.3)

        axs[0, 0].set_title('$E$', fontsize=16, fontweight='bold')
        axs[0, 1].set_title('$U$', fontsize=16, fontweight='bold')
        axs[1, 0].set_title('$\Gamma_n$', fontsize=16, fontweight='bold')
        axs[1, 1].set_title('$\Gamma_c$', fontsize=16, fontweight='bold')
        axs[1, 0].set(xlabel='time')
        axs[1, 1].set(xlabel='time')

    # Format the legend to the right of the plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.83)

    if foldername:
        filename = '/physics.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()


def rolling_stats(data: jnp.ndarray, window_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute rolling mean and std of the data with convolution.
    
    Args:
        data (jnp.ndarray): time series data to compute rolling statistics (e.g. energy)
        window_size (int): Size of the window for the convolution

    Returns:
        rolling_mean (jnp.ndarray): Rolling mean of the data array
        rolling_std (jnp.ndarray): Rolling standard deviation of the data array

    """
    mean = jnp.mean(data)
    rolling_mean = jnp.convolve(data, jnp.ones(window_size), mode='valid') / window_size
    rolling_std = jnp.sqrt(jnp.convolve((data - mean)**2, jnp.ones(window_size), mode='valid') / window_size)

    return rolling_mean, rolling_std
    
def plot_pdf_single(seq_dict: Dict, foldername: str=None):
    """
    Plot the bivariate time-PDF of phi and density of the predicted and true sequences for each model,
    centered around the center pixels of the domain, to mostly include spatial variability.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        foldername (str): Folder to save the plots

    """
    seq_dict = seq_dict.copy()

    _, ax = plt.subplots(1, len(seq_dict)-1, figsize=(15, 5))
    for i, key in enumerate(seq_dict.keys()):
        if i==0: # skip DNS
            continue
        
        #Get the center pixels of the images and flatten
        phi = np.array(seq_dict[key]['pred'][5:, 60:70, 60:70, 0].flatten())
        n = np.array(seq_dict[key]['pred'][5:, 60:70, 60:70, 1].flatten())
        phi_gt = np.array(seq_dict[key]['DNS'][5:, 60:70, 60:70, 0].flatten())
        n_gt = np.array(seq_dict[key]['DNS'][5:, 60:70, 60:70, 1].flatten())

        # Plot the bivariate time-PDF
        sns.kdeplot(x=phi, y=n, ax=ax[i-1], color='tab:orange', label='FNO')
        sns.kdeplot(x=phi_gt, y=n_gt, ax=ax[i-1], color='tab:blue', label='DNS')
    ax[0].set(xlabel=r'$\phi$', ylabel=r'PDF $\phi$')
    plt.legend(title='Model', title_fontsize='large')

    plt.tight_layout()
    if foldername:
        filename = '/pdf_single.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    
def plot_pdf_images(
        seq_dict: Dict, 
        param_dict: Dict, 
        in_images: int, 
        foldername: str=None
) -> None:
    """
    Plot the spatial PDF of phi and density of the predicted and true sequences 
    for each model at different times.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        in_images (int): Number of input images to the model
        foldername (str): Folder to save the plots
    
    """
    seq_dict = seq_dict.copy()

    M = seq_dict['DNS']['DNS'].shape[0] # time length
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    #snapshots considered for the plot: 10, 20 and the last one
    snaps = [in_images+9, in_images+19, M-1]

    fig, axs = plt.subplots(2, len(snaps), figsize=(15, 5))
    for i, key in enumerate(seq_dict.keys()):
        # Get time array
        dt = param_dict[key]['dt']
        times = jnp.arange(0, M*dt, dt)
        times = times - (in_images-1)*dt

        # Get the spatial PDF of phi and density for each model at each snapshot
        for j, snap in enumerate(snaps):
            phi = np.array(seq_dict[key]['pred'][snap, :, :, 0].flatten())
            n = np.array(seq_dict[key]['pred'][snap, :, :, 1].flatten())
            sns.kdeplot(phi, ax=axs[0, j], label=key)
            sns.kdeplot(n, ax=axs[1, j], label=key)

            axs[0, j].set_ylabel('' if j > 0 else 'spatial PDF $\phi$', fontsize=15)
            axs[1, j].set_ylabel('' if j > 0 else 'spatial PDF $n$', fontsize=15)
            axs[0, j].set_title(f"t = {times[snap]:.2f} - {snap-in_images+1}$\\delta t_{{NN}}$")
            axs[0, j].lines[i].set_dashes(line_styles[i])
            axs[1, j].lines[i].set_dashes(line_styles[i])
    # Set legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(right=0.85)
    fig.legend(handles, labels, loc='upper right', title='Model', title_fontsize='large')
            
    plt.tight_layout()
    if foldername:
        filename = '/pdf_images.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()


def plot_rsquared(
        seq_dict: Dict, 
        param_dict: Dict, 
        in_images: int, 
        foldername: str=None
) -> None:
    """
    Plot the R-squared value of the predicted and true sequences over time to display the quality 
    of the predictions.
    NOTE: Due to the error accumulation this is not very insightful for long time series,
    and has not been used in the report.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        in_images (int): Number of input images to the model
        foldername (str): Folder to save the plots
    
    """
    seq_dict = seq_dict.copy()

    _, ax = plt.subplots()
    for _, key in enumerate(seq_dict.keys()):
        # time settings
        dt = param_dict[key]['dt']
        times = jnp.round(jnp.linspace(0, (seq_dict['DNS']['DNS'].shape[0]-1)*dt, seq_dict['DNS']['DNS'].shape[0])-1, 2)
        times = times[in_images:] - (in_images)*dt

        # Initialize R-squared array and iterate over the time steps to fill
        Rsquared = jnp.zeros((seq_dict['DNS']['DNS'].shape[0]-in_images-1))
        for t in range(in_images, seq_dict[key]['pred'].shape[0]):
            seq = seq_dict[key]['pred'][t, :, :, :]
            Rsquared = Rsquared.at[t-in_images].set(get_Rsquared(seq, seq_dict['DNS']['DNS'][t, :, :, :]))

        # truncate to same lenth
        maxval = jnp.min(jnp.array([Rsquared.shape[0], times.shape[0]])) 
        Rsquared = Rsquared[:maxval]
        times = times[:maxval]
        ax.plot(times, Rsquared, label=f'{key}')
    ax.set(xlabel='time', ylabel='HW Residue')
    ax.legend()

    plt.tight_layout()
    if foldername:
        filename = '/Rsquare.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

def get_Rsquared(pred: jnp.ndarray, gt:jnp.ndarray) -> jnp.ndarray:
    """
    Compute the R-squared value between the predicted and ground truth sequences.
    
    Args:
        pred (jnp.ndarray): Predicted sequence
        gt (jnp.ndarray): Ground truth sequence
    """
    numerator = jnp.sum((gt - pred)**2)
    denominator = jnp.sum((gt - jnp.mean(gt))**2)
    return 1 - numerator / denominator

def plot_autocorr(
        seq_dict: Dict, 
        param_dict: Dict, 
        in_images: int, 
        scan_length: int=None, 
        foldername: str=None
) -> None:
    """
    Plot the autocorrelation of the predicted and true sequences over time and display 
    the decorrelation time. Autocorrelation is calculated over 60 timesteps by default.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        in_images (int): Number of input images to the model
        scan_length (int): time length the autocorrelation is averaged over
        foldername (str): Folder to save the plots
    """
    seq_dict = seq_dict.copy()

    # Set up line styles for the different models
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    # If no scan length is given, take the whole sequence - 60 due to the window of 60 timesteps
    if scan_length is None:
        scan_length = seq_dict['DNS']['DNS'].shape[0] - 60
    
    # Set the threshold for the decorrelation time: Corr(tau_d) = 1/e
    threshold = 1 / jnp.exp(1)

    _, ax = plt.subplots()
    autos = {} # store the autocorrelations
    autos_gt = {} # store the autocorrelations of the DNS
    tau_dict = {} # store the decorrelation times

    # Iterate over all models
    for i, key in enumerate(seq_dict.keys()):
        # time settings and initialize arrays
        dt = param_dict[key]['dt']
        times = jnp.linspace(0, 60*dt, 60)
        autocorrelations = jnp.zeros((scan_length-in_images, times.shape[0]))
        autocorrelations_gt = jnp.zeros((scan_length-in_images, times.shape[0]))
        taus = []
        taus_gt = []

        # iterate over the time steps t to calculate the autocorrelation C(t)
        for t_start in range(in_images, scan_length):
            reference = seq_dict[key]['pred'][t_start, :, :, 0].ravel()
            reference_gt = seq_dict[key]['DNS'][t_start, :, :, 0].ravel()

            # slide over the autocorrelation window and get C(t+t') for each t'
            for t in range(t_start, t_start+60):
                seq = seq_dict[key]['pred'][t, :, :, 0].ravel()
                seq_gt = seq_dict[key]['DNS'][t, :, :, 0].ravel()

                # Calculate the autocorrelation
                autocorr = jnp.dot(seq, reference) / (jnp.linalg.norm(seq) * jnp.linalg.norm(reference))
                autocorr_gt = jnp.dot(seq_gt, reference_gt) / (jnp.linalg.norm(seq_gt) * jnp.linalg.norm(reference_gt))
                autocorrelations = autocorrelations.at[t_start-in_images, t-t_start].set(autocorr)
                autocorrelations_gt = autocorrelations_gt.at[t_start-in_images, t-t_start].set(autocorr_gt)

            # Get the decorrelation time - restrict to first 10t as it's lower than t=10 for our settings
            first_autocorrs = autocorrelations[t_start-in_images, :20]
            first_autocorrs_gt = autocorrelations_gt[t_start-in_images, :20]
            
            # if threshold is surpassed, get the decorrelation time as the argmin w.r.t. the threshold
            if jnp.any(first_autocorrs < threshold):
                tau_indx = jnp.argmin(jnp.abs(first_autocorrs-threshold))
                taus.append(times[tau_indx])
                taus_indx_gt = jnp.argmin(jnp.abs(first_autocorrs_gt-threshold))
                taus_gt.append(times[taus_indx_gt])
            else:
                taus.append(jnp.nan)
                taus_gt.append(jnp.nan)

        # set 0 values to nans -> ignore for mean calculation
        autocorrelations = jnp.where(autocorrelations == 0, jnp.nan, autocorrelations)
        autocorrelations_gt = jnp.where(autocorrelations_gt == 0, jnp.nan, autocorrelations_gt)

        # get mean, ignoring nans
        autocorrs = jnp.nanmean(autocorrelations, axis=0)
        autocorrs_gt = jnp.nanmean(autocorrelations_gt, axis=0)

        # store the results
        autos[key] = autocorrs
        autos_gt[key] = autocorrs_gt

        # get the mean and std of the decorrelation times
        taus = jnp.array(taus)
        taus_gt = jnp.array(taus_gt)
        taus_gt_mean = jnp.mean(taus_gt)
        taus_gt_std = jnp.std(taus_gt)
        tau_dict[key] = {
            'mean_gt': taus_gt_mean, 
            'std_gt': taus_gt_std, 
            'mean': jnp.mean(taus), 
            'std': jnp.std(taus)
            }
        if jnp.any(jnp.isnan(taus)):
            tau_label = rf'{key}: $\tau_d = -$'
        else:
            tau = jnp.mean(jnp.array(taus))
            tau_std = jnp.std(jnp.array(taus))
            tau_SE = tau_std / jnp.sqrt(len(taus))
            if jnp.isnan(tau_SE) or jnp.isnan(tau):
                tau_label = rf'{key}: $\tau_d = -$'
            else:
                tau_label = rf'{key}: $\tau_d = {tau:.2f}\pm{tau_SE:.2f}$'
        ax.plot(times, autocorrs, dashes=line_styles[i], label=tau_label)
    ax.axhline(y=threshold, color='black', linestyle='--')
    ax.text(
        x=-1,  
        y=threshold +0.02,  
        s=r"$1/e$",
        color="black",
        fontsize=13,
        clip_on=False
        )
    ax.set_xlabel(r'time lag $\tau$', fontsize=16)
    ax.set_ylabel(r'$\mathcal{C}(\tau)$', fontsize=16)
    ax.legend(title='Model', title_fontsize='large')

    plt.tight_layout()
    if foldername:
        filename = '/autocorr.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    
    # Additionally, plot the decorrelation times in case of different physical regimes
    set1 = cm.get_cmap("Set1")
    colors = [set1(i) for i in range(8)]

    _, ax = plt.subplots()
    for i, key in enumerate(autos.keys()):
        if key == 'DNS':
            continue
        label = rf'{key}: $\tau_d = {tau_dict[key]["mean"]:.2f}\pm{tau_dict[key]["std"]:.2f}$'
        if jnp.isnan(tau_dict[key]['mean']) or jnp.isnan(tau_dict[key]['std']):
            label = rf'{key}: $\tau_d = -$'
        ax.plot(times, autos[key], color=colors[i-1], dashes=line_styles[i], label=label)
        ax.plot(times, autos_gt[key], color=colors[i-1], linewidth=1, alpha=0.9, linestyle=':')
    ax.axhline(y=threshold, color='black', linestyle='--')
    ax.text(
        x=-1,
        y=threshold +0.02,
        s=r"$1/e$",
        color="black",
        fontsize=13,
        clip_on=False
        )
    ax.set_xlabel(r'time lag $\tau$', fontsize=16)
    ax.set_ylabel(r'$\mathcal{C}(\tau)$', fontsize=16)
    ax.legend(title='Configuration', title_fontsize='large')
    plt.tight_layout()
    if foldername:
        filename = '/autocorr_multiregime.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

def plot_POD(
        seq_dict: Dict, 
        param_dict: Dict, 
        dx: float, 
        in_images: int, 
        foldername: str=None
) -> None:
    """
    Plot the POD spectrum of the predicted and true sequences for each model for phi, density and vorticity.
    
    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        param_dict (Dict): Dictionary containing the parameters of each model
        dx (float): spatial resolution
        in_images (int): Number of input images to the model
        foldername (str): Folder to save the plots

    """
    seq_dict = seq_dict.copy()

    # Set up line styles for the different models
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    # Iterate over the different fields and plot the POD spectrum of phi, density and vorticity
    _, ax = plt.subplots(1,3, figsize=(15, 5))
    for i, key in enumerate(seq_dict.keys()):
        # set time settings
        dt = param_dict[key]['dt']
        times = jnp.round(jnp.linspace(0, seq_dict['DNS']['DNS'].shape[0]*dt, seq_dict['DNS']['DNS'].shape[0]), 2)
        times = times[in_images:] - (in_images)*dt

        # Get the fields
        phi = seq_dict[key]['pred'][in_images:, :, :, 0]
        n = seq_dict[key]['pred'][in_images:, :, :, 1]
        omega = periodic_laplace(phi, dx)

        # Compute the POD spectrum of each field
        S = get_POD(phi)
        ax[0].plot(S, dashes=line_styles[i], label=rf'{key}')
        S = get_POD(n)
        ax[1].plot(S, dashes=line_styles[i], label=rf'{key}')
        S = get_POD(omega)
        ax[2].plot(S, dashes=line_styles[i], label=rf'{key}')
    
    # Set labels and legend
    titles = [r'potential $\phi$', r'density $n$', r'vorticity $\Omega$']
    for i in range(3):
        ax[i].set(xlabel=r'POD rank $j$', ylabel=r'normalised singular value $\tilde{\sigma_j}$' if i==0 else '') #ylim=(0, 1000)
        ax[i].legend(title='Configuration', title_fontsize='large')
        ax[i].set_title(titles[i], fontsize=16)
    ax[0].set_ylim([0, 0.05])
    ax[1].set_ylim([0, 0.04])

    plt.tight_layout()
    if foldername:
        filename = '/POD.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()


def get_POD(field: jnp.ndarray, norm: bool=True) -> jnp.ndarray:
    """
    Compute POD and return the singular values only.

    Args:
        field (jnp.ndarray): field of shape (t,x,y)
        norm (bool): optionally normalize the singular values
    
    Returns:
        S (jnp.ndarray): singular values of the field

    """
    # Preprocess field to be of shape (t, x*y)
    # NOTE that this gives the same singular values as (x*y, t)
    field = field.reshape(field.shape[0], -1)

    # Compute SVD
    S = jnp.linalg.svd(field, full_matrices=False, compute_uv=False)
    if norm:
        S = S / jnp.sum(S)
    
    return S


def plot_Q_criterion(
        seq_dict: Dict, 
        dx: float, 
        in_images: int, 
        time_length: int, 
        foldername: str = None
) -> None: 
    """
    Plot spatio-temporal distribution of the Q-criterion for the predicted and DNS sequences.
    Additionally, plot snapshots of the Q-criterion for each model.

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        dx (float): grid spacing
        in_images (int): Number of input images to the model
        time_length (int): Length of the time series
        foldername (str): Folder to save the plots
    
    """
    seq_dict = seq_dict.copy()

    # Cut off the first in_images timesteps for all sequences
    seq_dict = {outer_key: {inner_key: value[in_images-1:, :, :, :] for inner_key, value in outer_value.items()}
                for outer_key, outer_value in seq_dict.items()}

    # Set up line styles for the different models
    line_styles = [(1,0.01), (5, 2,  1, 1), (5, 3), (3, 2), (1, 1)]

    Q_snaps = [] # store a snapshot of all models
    preds = {}
    gts = {}

    fig, ax = plt.subplots()
    for i, key in enumerate(seq_dict.keys()):
        # If no time length is given, take the whole sequence
        if time_length is None:
            time_length = seq_dict[key]['DNS'].shape[0]
        
        # Get Q-values
        Q = get_Q_value(seq_dict[key]['pred'][:, :, :, 0], dx)
        Q_gt = get_Q_value(seq_dict[key]['DNS'][:, :, :, 0], dx)

        # Store the Q-values
        preds[key] = Q
        gts[key] = Q_gt
        Q_snaps.append(Q[-1, ...])

        # Flatten the Q-values and plot the spatio-temporal distribution, taking only the 1-99 percentile
        Q_pdf = Q.flatten()
        Q_pdf = Q_pdf[jnp.logical_and(Q_pdf > jnp.percentile(Q_pdf, 1), Q_pdf < jnp.percentile(Q_pdf, 99))]
        sns.kdeplot(Q_pdf, ax=ax, label=key)

    ax.set(xlabel=r'$Q$', ylabel='spatio-temporal distribution of $Q$', yscale='log', ylim=(2e-2, 1), xlim=(-5.8, 2.5))
    #ylim=(1e-2, 1), xlim=(-7.5, 2.5)
    ax.legend(title='Model', title_fontsize='large')
    for i, _ in enumerate(seq_dict.keys()):
        ax.lines[i].set_dashes(line_styles[i])
    plt.tight_layout()
    if foldername:
        filename = '/Q_criterion.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    # plot snapshots

    # Set up the box and colorbar limits
    L = 2*jnp.pi / 0.15
    box = [-L/2, L/2, -L/2, L/2]

    Q_0 = jnp.sqrt(jnp.mean(jnp.square(Q_pdf)))
    Q_0 = jnp.sqrt(jnp.mean(jnp.square(Q_snaps[0])))
    vmin = -3*Q_0
    vmax = 3*Q_0

    fig, axs = plt.subplots(1, len(Q_snaps), figsize=(15, 5))
    for i, key in enumerate(seq_dict.keys()):
        im = axs[i].imshow(Q_snaps[i], cmap='seismic', vmin=vmin, vmax=vmax, extent=box)
        axs[i].grid(False)
        axs[i].set_title(key, fontsize=16)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=im.cmap(vmax//2), 
                   markersize=10, label="Strain dominated"
                   ),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=im.cmap(vmin//2), 
                   markersize=10, label="Vorticity dominated"
                   )
                ]
    axs[-1].legend(handles=handles, loc="lower right")

    # add colorbar relative to the last axis
    cax = axs[-1].inset_axes((1.05, 0.0, 0.08, 1.0))
    fig.colorbar(im, cax=cax, label='Q')
    
    plt.tight_layout()
    if foldername:
        filename = '/Q_criterion_snaps.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

    # Plot distribution of all models with DNS, in case of multiple models
    set1 = cm.get_cmap("Set1")
    colors = [set1(i) for i in range(8)]

    fig, ax = plt.subplots()
    for i, key in enumerate(seq_dict.keys()):
        if i == 0:
            continue
        # Get the Q-values
        Q = preds[key]
        Q_gt = gts[key]
        Q_pdf = Q.flatten()
        Q_pdf_gt = Q_gt.flatten()

        # Cut top and bottom 1% of values
        Q_pdf = Q_pdf[jnp.logical_and(Q_pdf > jnp.percentile(Q_pdf, 1), Q_pdf < jnp.percentile(Q_pdf, 99))]
        Q_pdf_gt = Q_pdf_gt[jnp.logical_and(Q_pdf_gt > jnp.percentile(Q_pdf_gt, 1), 
                                            Q_pdf_gt < jnp.percentile(Q_pdf_gt, 99))
                            ]
        sns.kdeplot(Q_pdf, ax=ax, color=colors[i-1], label=key)
        sns.kdeplot(Q_pdf_gt, ax=ax, color=colors[i-1], linewidth=1, linestyle=':', alpha=0.9)

    # Apply line styles, as each 2nd line is a DNS
    for i, line in enumerate(ax.lines):
        if i % 2 == 0:
            line.set_dashes(line_styles[(i+1) // 2 % len(line_styles)])
    ax.set(xlabel=r'$Q$', ylabel=r'spatio-temporal distribution of $Q$', 
           yscale='log', ylim=(2e-2, 1), xlim=(-6.8, 3)
           )
    ax.legend(title='Configuration', title_fontsize='large') #ylim=(2e-2, 1), xlim=(-6.8, 3)

    plt.tight_layout()
    if foldername:
        filename = '/Q_criterion_all.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()
    

def get_Q_value(phi: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute the Q criterion for a given field.

    Args:
        phi (jnp.ndarray): potential field (t, x, y)
        dx (float): grid spacing

    Returns:
        Q (jnp.ndarray): Q-field

    """
    # Compute the gradient of the field
    x = periodic_gradient(phi, dx, axis=-1)
    y = periodic_gradient(phi, dx, axis=-2)
    xy = periodic_gradient(y, dx, axis=-1)
    yx = periodic_gradient(x, dx, axis=-2)
    xx = periodic_gradient(x, dx, axis=-1)
    yy = periodic_gradient(y, dx, axis=-2)
    omega = periodic_laplace(phi, dx)

    s1 = -xy + yx
    s2 = xx - yy

    Q = 1/4 * (s1**2 +s2**2 - omega**2)

    return Q


def print_prediction_errors(
        state: ExtendedTrainState, 
        model_name: str, 
        sequence: jnp.ndarray, 
        in_images: int, 
        rms: float
) -> None:
    """
    Print the prediction errors for a model for 1, 10 and 20 timesteps ahead.

    Args:
        state (ExtendedTrainState): State of the model
        model_name (str): Name of the model
        sequence (jnp.ndarray): Sequence to compute the prediction errors on
        in_images (int): Number of input images to the model
        rms (float): Root mean square of the sequence

    """

    # Preprocess dataset to 100 shorter sequences 
    seq_list = split_into_shorter_sequences(sequence, 25)
    seq_list = jnp.array(seq_list)

    # randomly shuffle the lists
    shuffle_key = jax.random.PRNGKey(42)
    shuffled_seq_list = jax.random.permutation(shuffle_key, seq_list)
    seq_list = shuffled_seq_list[:100]

    # Get the prediction errors by vectorizing over the list
    loss1, loss2, loss3 = vmap(get_prediction_errors,
                               in_axes=(0, None, None, None))(seq_list, state, in_images, rms)
    print('LOSSES SHAPE', loss1.shape, loss2.shape, loss3.shape)
    loss1_m = jnp.mean(loss1)
    loss2_m = jnp.mean(loss2)
    loss3_m = jnp.mean(loss3)
    loss1_std = jnp.std(loss1) 
    loss2_std = jnp.std(loss2) 
    loss3_std = jnp.std(loss3) 

    print('-'*50)
    print(f'Prediction errors for {model_name}:')
    print(f'1 timestep ahead: {loss1_m:.3f} +/- {loss1_std:.3f}')
    print(f'10 timesteps ahead: {loss2_m:.3f} +/- {loss2_std:.3f}')
    print(f'20 timesteps ahead: {loss3_m:.3f} +/- {loss3_std:.3f}')
    print('-'*50)


def get_prediction_errors(
        short_seq: jnp.ndarray, 
        state: ExtendedTrainState, 
        in_images: int, 
        rms: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the prediction errors for a single sequence for 1, 10 and 20 timesteps ahead.

    Args:
        short_seq (jnp.ndarray): Sequence of 25 timesteps length
        state (ExtendedTrainState): State of the model
        in_images (int): Number of input images to the model
        rms (float): Root mean square of the sequence

    """
    # Recover the original sequence
    gt_seq = short_seq * rms

    # Initialize the prediction
    prediction = jnp.zeros_like(short_seq)
    norm_prediction = jnp.zeros_like(short_seq)
    prediction = prediction.at[:, :, :in_images, :].set(short_seq[:, :, :in_images, :])
    norm_prediction = norm_prediction.at[:, :, :in_images, :].set(gt_seq[:, :, :in_images, :])
    variables = {'params': state.params, 'batch_stats': state.batch_stats}

    # Run sequence
    for t in range(in_images, short_seq.shape[2]):
        input = prediction[:, :, t - in_images:t, :].copy()
        X = seq_to_X(input, 128) # fixed res at 128x128
        pred = state.apply_fn(variables, X, train=False)
        prediction = prediction.at[:, :, t, :].set(pred)
        norm_prediction = norm_prediction.at[:, :, t, :].set(pred * rms)

    # remove the first in_images
    gt_seq = gt_seq[:, :, in_images:, :]
    norm_prediction = norm_prediction[:, :, in_images:, :]

    # Get single losses
    gt1 = gt_seq[:, :, 0, :]
    pred1 = norm_prediction[:, :, 0, :]
    loss1 = get_rel_l2_loss(gt1, pred1)

    # Get multi-pred losses for 10 timesteps
    gt10 = gt_seq[:, :, :10, :]
    pred10 = norm_prediction[:, :, :10, :]
    loss2 = get_rel_l2_loss(gt10, pred10)

    # get multi-pred losses for 20 timesteps
    gt20 = gt_seq[:, :, :20, :]
    pred20 = norm_prediction[:, :, :20, :]
    loss3 = get_rel_l2_loss(gt20, pred20)

    return loss1, loss2, loss3


def POD_analysis(
        seq_dict: Dict, 
        dx: float, 
        snap: int, 
        variable: str='phi', 
        foldername: str=None
) -> None:
    """
    POD decomposition and display of the POD modes of a given snapshot

    Args:
        seq_dict (Dict): Dictionary containing the predicted and true sequences for each model
        dx (float): spatial resolution
        snap (int): Snapshot to display
        variable (str): Variable to perform the POD analysis on
        foldername (str): Folder to save the plots
    
    """
    seq_dict = seq_dict.copy()
    cmap = get_cmap()
    variable = variable.lower()
    try:
        vars = {'phi': 0, 'n': 1, 'omega': 2}[variable]
    except KeyError:
        raise ValueError('Variable must be one of "phi", "n" or "omega"')
    
    # If the variable is omega, compute it from phi, else take the variable directly
    if vars == 2:
        y_seq = periodic_laplace(seq_dict['DNS']['DNS'][:, :, :, 0], dx)
    else:
        y_seq = seq_dict['DNS']['DNS'][:, :, :, vars]

    # Get the POD field of shape (x*y, t) and compute the SVD
    y_f = y_seq.transpose(1, 2, 0).reshape(y_seq.shape[1] * y_seq.shape[2], -1)
    U_gt, S_gt, V_gt = jnp.linalg.svd(y_f, full_matrices=True)
    gt_image = y_seq[:, :, snap]

    # Set the box size
    L = 2*jnp.pi / 0.15
    box = [-L/2, L/2, -L/2, L/2]

    # POD modes to display
    PODs = [0, 4, 9, 24, 49]

    _, axs = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for i, pod in enumerate(PODs):
        # Recover the field at t=snap of given basis vector
        f = S_gt[pod] * U_gt[:, pod] * V_gt[pod, snap]
        f = f.reshape(y_seq.shape[1], y_seq.shape[2])

        # plot the POD mode recreated field
        im = axs[i].imshow(f, cmap=cmap, extent=box)
        axs[i].grid(False)
        axs[i].set_title(f'POD mode {pod+1}')
        cbar = plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if foldername:
        filename = f'/POD_modes_final_{variable}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()
    
    # save the original field
    _, ax = plt.subplots()
    im = ax.imshow(gt_image, cmap=cmap, extent=box)
    ax.grid(False)
    ax.set_title(r'original $\phi$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    if foldername:
        filename = f'/POD_modes_final_{variable}_single.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    
    # plot the singular value per rank and at each POD value in PODs red dashed line with value
    _, ax = plt.subplots()
    ax.plot(S_gt / jnp.sum(S_gt))
    for pod in PODs:
    # Plot a larger red dot at each POD point
        ax.scatter(pod, (S_gt[pod] / jnp.sum(S_gt)), color='tab:red', s=100, zorder=5)
        ax.text(x=pod+6, y=(S_gt[pod] / jnp.sum(S_gt)) + 0.0015, s=f'{pod+1}', 
                fontsize=10, color='black', ha='center', rotation=0
                )
    ax.set(xlabel=r'POD rank $j$', ylabel=r'$\tilde{\sigma_j}$')
    ax.legend()

    plt.tight_layout()
    if foldername:
        filename = f'/POD_modes_singular_values_{variable}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()


    



    



