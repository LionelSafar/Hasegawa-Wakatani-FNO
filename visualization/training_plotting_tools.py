import os
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.physical_quantities import periodic_laplace, periodic_gradient

# This file contains the plotting functions used for immediate visualisation of the results after 
# training a model.

def plot_physics(
          gt_stats: Dict, 
          pred_stats: Dict, 
          foldername: str = None, 
          trainflag: str = None, 
          id: str = None
) -> None:
    """
    Plot the physics quantities for the predicted and true data.

    Args:
    - gt_stats: dictionary containing the ground truth statistics
    - pred_stats: dictionary containing the predicted statistics
    - foldername: folder to save the plot
    - trainflag: flag to indicate end of training as vertical line in the plots
    - id: id to append to the filename
    
    """
    # Get time and the flag index
    time = pred_stats['time']
    if trainflag:
        flag_index = int(trainflag * len(time))

    # Create the subplots
    _, ax = plt.subplots(2, 4, figsize=(15, 10), sharex=True)

    # Energy
    mean_pred = jnp.mean(pred_stats['energy'])
    std_pred = jnp.std(pred_stats['energy'])
    mean_gt = jnp.mean(gt_stats['energy'])
    std_gt = jnp.std(gt_stats['energy'])

    ax[0, 0].plot(time, pred_stats['energy'], color='tab:orange',
                  label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[0, 0].plot(time, gt_stats['energy'], color='tab:blue',
                  label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[0, 0].set_title('Energy', fontsize=16, fontweight='bold')
    if trainflag:
            ax[0, 0].axvline(time[flag_index], color='black', linestyle='--')
            ax[0, 0].text(time[flag_index], ax[0, 0].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[0, 0].legend()

    # Enstrophy
    mean_pred = jnp.mean(pred_stats['enstrophy'])
    std_pred = jnp.std(pred_stats['enstrophy'])
    mean_gt = jnp.mean(gt_stats['enstrophy'])
    std_gt = jnp.std(gt_stats['enstrophy'])

    ax[1, 0].plot(time, pred_stats['enstrophy'], color='tab:orange',
                  label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[1, 0].plot(time, gt_stats['enstrophy'], color='tab:blue',
                  label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[1, 0].set_title('Enstrophy', fontsize=16, fontweight='bold')
    ax[1, 0].set(xlabel='time')
    if trainflag:
            ax[1, 0].axvline(time[flag_index], color='black', linestyle='--')
            ax[1, 0].text(time[flag_index], ax[1, 0].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[1, 0].legend()

    # Gamma_n
    mean_pred = jnp.mean(pred_stats['gamma_n'])
    std_pred = jnp.std(pred_stats['gamma_n'])
    mean_gt = jnp.mean(gt_stats['gamma_n'])
    std_gt = jnp.std(gt_stats['gamma_n'])

    ax[0, 1].plot(time, pred_stats['gamma_n'], color='tab:orange',
                  label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[0, 1].plot(time, gt_stats['gamma_n'], color='tab:blue',
                  label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[0, 1].set_title('$\Gamma_n$', fontsize=16, fontweight='bold')
    if trainflag:
            ax[0, 1].axvline(time[flag_index], color='black', linestyle='--')
            ax[0, 1].text(time[flag_index], ax[0, 1].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[0, 1].legend()

    # Gamma_c
    mean_pred = jnp.mean(pred_stats['gamma_c'])
    std_pred = jnp.std(pred_stats['gamma_c'])
    mean_gt = jnp.mean(gt_stats['gamma_c'])
    std_gt = jnp.std(gt_stats['gamma_c'])

    ax[1, 1].plot(time, pred_stats['gamma_c'], color='tab:orange', 
                  label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[1, 1].plot(time, gt_stats['gamma_c'], color='tab:blue', 
                  label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[1, 1].set_title('$\Gamma_c$', fontsize=16, fontweight='bold')
    ax[1, 1].set(xlabel='time')
    if trainflag:
            ax[1, 1].axvline(time[flag_index], color='black', linestyle='--')
            ax[1, 1].text(time[flag_index], ax[1, 1].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[1, 1].legend()

    # DE
    mean_pred = jnp.mean(pred_stats['DE'])
    std_pred = jnp.std(pred_stats['DE'])
    mean_gt = jnp.mean(gt_stats['DE'])
    std_gt = jnp.std(gt_stats['DE'])

    ax[0, 2].plot(time, pred_stats['DE'], color='tab:orange',
                    label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[0, 2].plot(time, gt_stats['DE'], color='tab:blue',
                    label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[0, 2].set_title('DE', fontsize=16, fontweight='bold')
    if trainflag:
            ax[0, 2].axvline(time[flag_index], color='black', linestyle='--')
            ax[0, 2].text(time[flag_index], ax[0, 2].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[0, 2].legend()

    # DU
    if 'DU' in pred_stats: # vorticity case
        mean_pred = jnp.mean(pred_stats['DU'])
        std_pred = jnp.std(pred_stats['DU'])
        mean_gt = jnp.mean(gt_stats['DU'])
        std_gt = jnp.std(gt_stats['DU'])

        ax[1, 2].plot(time, pred_stats['DU'], color='tab:orange',
                        label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
        ax[1, 2].plot(time, gt_stats['DU'], color='tab:blue',
                        label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
        ax[1, 2].set_title('DU', fontsize=16, fontweight='bold')
        ax[1, 2].set(xlabel='time')
        if trainflag:
            ax[1, 2].axvline(time[flag_index], color='black', linestyle='--')
            ax[1, 2].text(time[flag_index], ax[1, 2].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
        ax[1, 2].legend()
    else:
        ax[1, 2].axis('off') # hide the subplot if not present

    # Potential
    mean_pred = jnp.mean(pred_stats['phi'])
    std_pred = jnp.std(pred_stats['phi'])
    mean_gt = jnp.mean(gt_stats['phi'])
    std_gt = jnp.std(gt_stats['phi'])

    ax[0, 3].plot(time, pred_stats['phi'], color='tab:orange',
                    label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[0, 3].plot(time, gt_stats['phi'], color='tab:blue',
                    label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[0, 3].set_title('Potential', fontsize=16, fontweight='bold')
    if trainflag:
            ax[0, 3].axvline(time[flag_index], color='black', linestyle='--')
            ax[0, 3].text(time[flag_index], ax[0, 3].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[0, 3].legend()

    # Density
    mean_pred = jnp.mean(pred_stats['n'])
    std_pred = jnp.std(pred_stats['n'])
    mean_gt = jnp.mean(gt_stats['n'])
    std_gt = jnp.std(gt_stats['n'])

    ax[1, 3].plot(time, pred_stats['n'], color='tab:orange',
                    label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
    ax[1, 3].plot(time, gt_stats['n'], color='tab:blue',
                    label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
    ax[1, 3].set_title('Density', fontsize=16, fontweight='bold')
    ax[1, 3].set(xlabel='time')
    if trainflag:
            ax[1, 3].axvline(time[flag_index], color='black', linestyle='--')
            ax[1, 3].text(time[flag_index], ax[1, 3].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
    ax[1, 3].legend()
    
    plt.tight_layout() 

    # Save the figure and show it if DISPLAY is set
    if foldername:
        if id:
            filename = f'/physics_{id}.pdf'
        else:
            filename = '/physics.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f'Physics plot saved at {foldername + filename}')
    if os.environ.get('DISPLAY', ''):
        plt.show()
    
    # enstrophy from omega as separate plot
    if 'enstrophy_omega' in pred_stats:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

        mean_pred = jnp.mean(pred_stats['enstrophy_omega'])
        std_pred = jnp.std(pred_stats['enstrophy_omega'])
        mean_gt = jnp.mean(gt_stats['enstrophy_omega'])
        std_gt = jnp.std(gt_stats['enstrophy_omega'])

        ax.plot(time, pred_stats['enstrophy_omega'], color='tab:orange',
                    label=f'prediction ({mean_pred:.2f}±{std_pred:.2f})')
        ax.plot(time, gt_stats['enstrophy_omega'], color='tab:blue',
                    label=f'ground truth ({mean_gt:.2f}±{std_gt:.2f})')
        ax.set_title('Enstrophy from Vorticity', fontsize=16, fontweight='bold')
        ax.set(xlabel='time')
        ax.legend()

        plt.tight_layout()
        if foldername:
            filename = '/enstrophy_omega.pdf'
            plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
        if os.environ.get('DISPLAY', ''):
            plt.show()


def plot_loss_with_time(
          y: jnp.ndarray, 
          pred: jnp.ndarray, 
          dt: float, 
          inc_vorticity: bool=True, 
          foldername: str = None, 
          trainflag: float = None, 
          id: str = None
) -> None:
    """
    Plot the relative L2 loss with time for the predicted and true data for each field

    Args:
        y (jnp.ndarray): DNS sequence
        pred (jnp.ndarray): predicted sequence
        dt (float): time step
        inc_vorticity (bool): whether vorticity should be included in the plot
        foldername (str): folder to save the plot
        trainflag (float): flag to indicate end of training as vertical line in the plots
        id (str): id to append to the filename

    """
    if inc_vorticity:
        variables = ['phi', 'density', 'omega']
    else:
        variables = ['phi', 'density']
    
    basesize = 5 # base size for the figure
    N = y.shape[2] # number of time steps

    if trainflag:
        train_indx = int(trainflag * N)
    times = jnp.linspace(0, N*dt, N)

    # Create the subplots
    _, ax = plt.subplots(1, len(variables), figsize=(basesize + 2.5*(len(variables)-1), 5))
    for i, variable in enumerate(variables):
        # Calculate and plot the loss
        num = jnp.linalg.norm(pred[:, :, :, i] - y[:, :, :, i], axis=(0, 1))
        denom = jnp.linalg.norm(y[:, :, :, i], axis=(0, 1))
        loss = num / denom
        ax[i].plot(times, loss)
        if trainflag:
            ax[i].axvline(times[train_indx], color='black', linestyle='--')
            ax[i].text(times[train_indx], ax[i].get_ylim()[0] - 0.05, 'End of Training', 
           color='black', ha='center', va='top', fontsize=10)
        ax[i].set(title=variable, xlabel='Time (s)', ylabel='MSE Loss')
        ax[i].grid(True)

    plt.tight_layout()
    if foldername:
        if id:
            filename = f'/time_losses_{id}.pdf'
        else:
            filename = f'/time_losses.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()

def plot_snapshot(
          y: jnp.ndarray, 
          pred: jnp.ndarray, 
          variable: str, 
          n_images: int, 
          dt: float, 
          foldername: str = None
) -> None:
    """
    Plot snapshots of the predicted and true data for a given variable evenly spaced over the time domain.
    
    Args:
        y (jnp.ndarray): DNS sequence
        pred (jnp.ndarray): predicted sequence
        variable (str): variable to plot ('phi', 'omega', 'density')
        n_images (int): number of images to plot
        dt (float): time step
        foldername (str): folder to save the plot
    
    """
    # Get the channel index
    variable = variable.lower()
    try:
        channel = {'phi': 0, 'density': 1, 'omega': 2}[variable]
    except KeyError:
        raise ValueError('variable must be one of phi, omega, density')
    
    # Get the time indices for the snapshots and the time array
    snap_indices = jnp.linspace(0, y.shape[2]-2, n_images, dtype=jnp.int32)
    N = y.shape[2]
    times = jnp.arange(0, N*dt, dt)

    # Create the subplots
    base_size = 5
    fig, ax = plt.subplots(2, n_images, figsize=(base_size + 3*n_images, 5))
    for i in range(n_images):
        # Get MSE loss
        loss = jnp.mean(jnp.square(pred[:, :, snap_indices[i], channel] - y[:, :, snap_indices[i], channel]))
        time = times[snap_indices[i]]

        # Fix the vmin and vmax for the colorbar based on the ground truth
        vmin = jnp.min(y[:, :, snap_indices[i], channel])
        vmax = jnp.max(y[:, :, snap_indices[i], channel])

        ax[0, i].imshow(y[:, :, snap_indices[i], channel], cmap='seismic', vmin=vmin, vmax=vmax)
        ax[1, i].imshow(pred[:, :, snap_indices[i], channel], cmap='seismic', vmin=vmin, vmax=vmax)
        ax[0, i].set_title(f't={time}', fontstyle='italic')
        ax[1, i].set_title(f'MSE: {loss:.2}', fontsize=10, fontstyle='italic')
        ax[0, i].axis('off')
        ax[1, i].axis('off')

        if i == 0:
            ax[0, i].set_ylabel('ground truth', fontweight='bold')
            ax[1, i].set_ylabel('prediction', fontweight='bold')
    
    plt.tight_layout()
    fig.suptitle(f'{variable.capitalize()} Snapshots', fontsize=14, fontweight='bold')
    if foldername:
        filename = f'/snapshots_{variable}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()
        

def plot_train_val_losses(metrics: dict, foldername: str = None) -> None:
    """
    Plot the training and validation losses.

    Args:
        metrics: dictionary containing the training and validation losses
        foldername: folder to save the plot

    """
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    val_aux_loss = metrics['val_aux_loss']

    _, ax = plt.subplots(1, 1)
    ax.plot(train_loss, label='training - curriculum')
    ax.plot(val_loss, label='validation - single pred.')
    ax.plot(val_aux_loss, label='validation - sequence pred.')
    ax.set(title='Training and Validation Loss', xlabel='Epoch', ylabel='Relative L2 loss')
    ax.set_ylim([0, 1])
    ax.legend(title='Loss')
    plt.tight_layout()

    #save figure
    if foldername:
        filename = 'train_val_losses.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()
    
    #Additionally plot auxiliary loss if it exists
    if 'aux_loss' in metrics:
        aux_loss = metrics['aux_loss']
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(aux_loss, label='aux loss')
        ax.set(title='Auxiliary Loss', xlabel='Epoch', ylabel='Relative l2 Loss')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if foldername:
            filename = 'aux_loss.pdf'
            plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
        if os.environ.get('DISPLAY', ''):
            plt.show()


def plot_snaps_and_grad(
        y: jnp.ndarray, 
        pred: jnp.ndarray, 
        variable: str, 
        n_images: int, 
        dt: float, 
        foldername: str = None
)->None:
    """
    Plot and save snapshots of the predicted and true data and their gradients up to second order.

    Args:
        y: DNS data of shape (Nx, Ny, Nt, features)
        pred: predicted data of shape (Nx, Ny, Nt, features)
        variable: variable to plot ('phi', 'omega', 'density')
        n_images: number of images to plot

    """
    variable = variable.lower()
    try:
        channel = {'phi': 0, 'density': 1, 'omega': 2}[variable]
    except KeyError:
        raise ValueError('variable must be one of phi, omega, density')
    
    # Only show last image of the 5 input images -> Will start at DNS image in both cases
    snap_indices = jnp.arange(4, 4+n_images, dtype=jnp.int32)
    N = y.shape[2]
    times = jnp.arange(0, N*dt, dt)

    base_size = 5
    fig, ax = plt.subplots(6, n_images, figsize=(base_size + 4*n_images, 30))
    for i in range(n_images):
        # Get loss
        num = jnp.linalg.norm(pred[:, :, snap_indices[i], channel] - y[:, :, snap_indices[i], channel])
        denom = jnp.linalg.norm(y[:, :, snap_indices[i], channel])
        loss = num / denom

        # Get time and set vmin and vmax for the colorbbar as same for both ground truth and prediction
        time = times[snap_indices[i]]
        vmin = jnp.min(y[:, :, snap_indices[i], channel])
        vmax = jnp.max(y[:, :, snap_indices[i], channel])
        square_y = jnp.mean(jnp.square(y[:, :, snap_indices[i], channel]))
        square_pred = jnp.mean(jnp.square(pred[:, :, snap_indices[i], channel]))
        
        # Plot ground truth
        im_y = ax[0, i].imshow(y[:, :, snap_indices[i], channel], cmap='seismic', vmin=vmin, vmax=vmax)
        cbar_y = plt.colorbar(im_y, ax=ax[0, i], orientation='vertical', fraction=0.046, pad=0.04)
        cbar_y.set_label('')

        # Plot prediction
        im_pred = ax[1, i].imshow(pred[:, :, snap_indices[i], channel], cmap='seismic', vmin=vmin, vmax=vmax)
        cbar_pred = plt.colorbar(im_pred, ax=ax[1, i], orientation='vertical', fraction=0.046, pad=0.04)
        cbar_pred.set_label('')

        # Set titles
        ax[0, i].set_title(f't={time} mean^2 {square_y}', fontsize=12, fontstyle='italic')
        ax[1, i].set_title(f'L2: {loss:.2f} mean^2 {square_pred}', fontsize=12, fontstyle='italic')

        # Remove axes
        ax[0, i].axis('off')
        ax[1, i].axis('off')

        if i == 0:
            ax[0, i].set_ylabel('ground truth', fontweight='bold')
            ax[1, i].set_ylabel('prediction', fontweight='bold')

        # Get gradients
        dx = 2*jnp.pi / 0.15 / 128
        grad_y = periodic_gradient(y[:, :, snap_indices[i], channel], dx, axis=-2)
        grad_pred = periodic_gradient(pred[:, :, snap_indices[i], channel], dx, axis=-2)
        grad_loss = jnp.linalg.norm(grad_pred - grad_y) / jnp.linalg.norm(grad_y) 
        grad_mean = jnp.mean(jnp.square(grad_y))
        grad_mean_pred = jnp.mean(jnp.square(grad_pred))

        # Fix vmin and vmax for gradients
        vmin = jnp.min(grad_y)
        vmax = jnp.max(grad_y)

        # Plot ground truth
        im_grad_y = ax[2, i].imshow(grad_y, cmap='seismic', vmin=vmin, vmax=vmax)
        cbar_grad_y = plt.colorbar(im_grad_y, ax=ax[2, i], orientation='vertical', fraction=0.046, pad=0.04)

        # Plot prediction
        im_grad_pred = ax[3, i].imshow(grad_pred, cmap='seismic', vmin=vmin, vmax=vmax)
        cbar_grad_pred = plt.colorbar(im_grad_pred, ax=ax[3, i], orientation='vertical', fraction=0.046, pad=0.04)

        # Set titles
        ax[2, i].set_title(f'mean{grad_mean:.2f}', fontsize=14, fontstyle='italic')
        ax[3, i].set_title(f'grad L2: {grad_loss:.2f}, mean{grad_mean_pred:.2f}', fontsize=14, fontstyle='italic')

        # Remove axes 
        ax[2, i].axis('off')
        ax[3, i].axis('off')

        if i == 0:
            ax[2, i].set_ylabel('gt grad', fontweight='bold')
            ax[3, i].set_ylabel('pred grad', fontweight='bold')

        # Get second gradients
        grad2_y = periodic_laplace(y[:, :, snap_indices[i], channel], dx)
        grad2_pred = periodic_laplace(pred[:, :, snap_indices[i], channel], dx)
        grad2_loss = jnp.linalg.norm(grad2_pred - grad2_y) / jnp.linalg.norm(grad2_y)
        grad2_mean = jnp.mean(jnp.square(grad2_y))
        grad2_mean_pred = jnp.mean(jnp.square(grad2_pred))

        # Plot gradients
        vmin = jnp.min(grad2_y)
        vmax = jnp.max(grad2_y)

        # Plot ground truth
        im_grad2_y = ax[4, i].imshow(grad2_y, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar(im_grad2_y, ax=ax[4, i], orientation='vertical', fraction=0.046, pad=0.04)

        # Plot prediction
        im_grad2_pred = ax[5, i].imshow(grad2_pred, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar(im_grad2_pred, ax=ax[5, i], orientation='vertical', fraction=0.046, pad=0.04)

        # Set titles
        ax[4, i].set_title(f'mean{grad2_mean:.2f}', fontsize=14, fontstyle='italic')
        ax[5, i].set_title(f'loss:{grad2_loss:.2f} mean{grad2_mean_pred:.2f}', fontsize=14, fontstyle='italic')

        # Remove axes
        ax[4, i].axis('off')
        ax[5, i].axis('off')

        if i == 0:
            ax[4, i].set_ylabel('gt laplace', fontweight='bold')
            ax[5, i].set_ylabel('pred laplace', fontweight='bold')

    plt.tight_layout()
    fig.suptitle(f'{variable.capitalize()} Snapshots', fontsize=14, fontweight='bold')
    if foldername:
        filename = f'/grad_and_snaps_{variable}.pdf'
        plt.savefig(foldername + filename, format='pdf', dpi=300, bbox_inches='tight')
    if os.environ.get('DISPLAY', ''):
        plt.show()