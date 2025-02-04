import argparse
import os
import re

import jax
import jax.numpy as jnp
import matplotlib as mpl

from utils.data_handling import SequenceDataset, get_sample
from utils.model_checkpointer import load_model, get_model_type
from utils.run_sequence import run_sequence

from visualization.posterior_plotting_tools import *

# This script is used to evaluate multiple neural network models and compare their predictions.
# It allows for:
# - comparison of multiple different models
# - comparison of different versions of the same model
# - comparison of different models trained on different datasets e.g. different phyiscal driive (kappa, nu)
# - comparison of different timestep between snapshots (might need adjustments in the code)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate multiple neural network models.")
    parser.add_argument('--model_paths', nargs='+', required=True, 
                        help="List of paths to trained model files (separated by spaces)."
                        )
    parser.add_argument('--model_version', type=str, default='final',
                        help="Version of the model to evaluate - 'final' or 'best'."
                        )
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the data file -- "
                        )
    parser.add_argument('--sequence_length', type=int, default=None,
                        help="Length of the sequence to evaluate. If None, use full sequence."
                        )
    parser.add_argument('--model_list', nargs='+', default=None,
                        help="List of model names to use for plotting. If None, use model type as default. \
                        Must be same length and order as model paths."
                        )
    parser.add_argument('--Data_list', nargs='+', default=None,
                        help="List of data names to use for plotting. If None, use data type as default. \
                        Must be same length and order as model paths."
                        )
    parser.add_argument('--sequence_length_time', type=int, default=None,
                        help="Length of the sequence in time to evaluate. If None, use full sequence."
                        )
    args = parser.parse_args()

    model_paths = args.model_paths
    model_version = args.model_version
    sequence_length = args.sequence_length
    model_list = args.model_list
    data_list = args.Data_list
    sequence_length_time = args.sequence_length_time

    # Validate inputs
    if model_version not in ['final', 'best']:
        raise ValueError("Invalid model version. Must be 'final' or 'best'.")
    if len(model_paths) == 0:
        raise ValueError("At least one model path must be provided.")
    print(50*'-')
    print(f'SELECTED {len(model_paths)} MODELS FOR EVALUATION')
    print(50*'-')
    print(f'MODEL VERSION: {model_version}')

    # Set plotting style, use ggplot
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

    # Get the save name of the model based on version selection
    folders = os.listdir(model_paths[0]) # Assume uniform structure among all model paths
    for folder in folders:
        if model_version in folder:
            save_name = folder
            break

    # Get list of all models:
    models = []
    for model_path in model_paths:
        # If model is final or best, do not use save_name
        if 'final' in model_path or 'best' in model_path:
            model = load_model(model_path, '')
        else:
            model = load_model(model_path, save_name)
        models.append(model)

    # Get physical parameters that are shared among all models
    resolution = int(re.findall(r'(\d+)res', model_paths[0])[0])
    print('RESOLUTION:', resolution)
    in_images = int(re.findall(r'(\d+)nimg', model_paths[0])[0])
    print('IN IMAGES:', in_images)
    t_start = int(re.findall(r'st(\d+)', model_paths[0])[0])
    print('T START:', t_start)
    c1 = float(re.findall(r'c1(\d+)', model_paths[0])[0])
    print('C1:', c1)
    dx = 2*jnp.pi / resolution / 0.15

    print(50*'-')
    print(f'EVALUATING MODELS')
    print(50*'-')
    seq_dict = {}
    param_dict = {}

    # iterate over all models and save their parameters and predictions
    for i, model in enumerate(models):
        # Get model structure, kappa, nu, dt
        model_type = get_model_type(model_paths[i])
        y_diff = ("diffy" in model_paths[i])
        timestep = int(re.findall(r'(\d+)t', model_paths[i])[0])
        print('TIMESTEP:', timestep)
        kappa = float(re.findall(r'k(\d+)', model_paths[i])[0])
        if kappa == 0:
            kappa = 0.5
        else:
            kappa = float(kappa)
        print('kappa:', kappa)
        nu = float(re.findall(r'nu([0-9e\.-]+)', model_paths[i])[0])
        print('NU:', nu)
        dt = 0.01 * 10 * timestep # Assume fixed simulation timestep and downsampling factor

        # Get folder path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(current_dir, '..', 'Data/'))
        simulation_settings = f'k{kappa}_c1{int(c1)}_nu{nu}'
        folder_path = os.path.join(path, f'{simulation_settings}/')
        folder_path = os.path.join(folder_path, f'model_figures/')
        os.makedirs(folder_path, exist_ok=True)
        key = jax.random.PRNGKey(0)


        if data_list: # If list is given, take corresponding dataset for each model
                data = SequenceDataset(data_list[i])
        else: # Otherwise, use the same dataset for all models
            data = SequenceDataset(args.data_path)
        
        # Get a sample sequence
        sequence = get_sample(key, data, sequential=True, 
                                in_images=in_images,
                                resolution=resolution,
                                y_diff=y_diff)
        
        # Optional sequence length truncation based on index
        if sequence_length:
            sequence = sequence[:, :, :sequence_length, :]

        # Optional sequence length truncation based on physical time
        if sequence_length_time:
            time_index = int(sequence_length_time // dt)
            sequence = sequence[:, :, :time_index, :]
        
        # Run the sequence through the model
        y, pred = run_sequence(sequence, model, data.get_rms(),
                               in_images=in_images, 
                               resolution=resolution,
                               y_diff=y_diff
                               )
        print_prediction_errors(model, 
                                model_list[i] if model_list else model_type, 
                                y, 
                                in_images, 
                                data.get_rms()
                                )

        # transpose to f(t, x, y, {phi, n})
        y = y.transpose(2, 0, 1, 3)
        pred = pred.transpose(2, 0, 1, 3)

        # Add DNS as first entry of the dictionary
        if i == 0:
            param_dict['DNS'] = {'kappa': kappa, 'nu': nu, 'dt': dt}
            seq_dict['DNS'] = {'DNS': y, 'pred': y} # to match the structure of the model dict 

        # Add models to the dictionary
        if model_list:
            seq_dict[model_list[i]] = {'DNS': y, 'pred': pred}
            param_dict[model_list[i]] = {'kappa': kappa, 'nu': nu, 'dt': dt}

        # If no model list is given, use model type as default
        else:
            seq_dict[model_type] = {'DNS': y, 'pred': pred}
            param_dict[model_type] = {'kappa': kappa, 'nu': nu, 'dt': dt}

    # Plot the model comparisons - length has to be adapted in the code based on needs.. 

    # Autocorrelation
    plot_autocorr(seq_dict, param_dict, in_images, scan_length=400, foldername=folder_path)

    # POD analysis of snapshot -- decompose field and display
    #POD_analysis(seq_dict, dx, snap=100, variable='phi', foldername=folder_path)
    #POD_analysis(seq_dict, dx, snap=100, variable='n', foldername=folder_path)
    #POD_analysis(seq_dict, dx, snap=100, variable='omega', foldername=folder_path)

    # Snapshots of fields, gradients and fourier fields
    plot_snaps(seq_dict, param_dict, 'phi', dx, in_images=in_images, foldername=folder_path)
    plot_snaps(seq_dict,  param_dict, 'n', dx, in_images=in_images, foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'phi', dx, in_images=in_images, 
                inc_error=True, mode='fft_magnitude', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images, 
                inc_error=True, mode='fft_magnitude', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'phi', dx, in_images=in_images,
                inc_error=True, mode='fft_phase', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images, 
                inc_error=True, mode='fft_phase', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images, 
                inc_error=True, mode='fft_real', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images,
                inc_error=True, mode='fft_imag', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'phi', dx, in_images=in_images, mode='gradient', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images, mode='gradient', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'phi', dx, in_images=in_images, mode='laplace', foldername=folder_path)
    plot_snaps(seq_dict, param_dict, 'n', dx, in_images=in_images, mode='laplace', foldername=folder_path)

    # Energy cascade
    plot_energy_cascade(seq_dict, dx, timelength=400, foldername=folder_path, id='last')

    # Error accumulation of fields, gradient fields and fourier fields
    plot_loss_with_time(seq_dict, param_dict, dx, in_images, mode='basic', foldername=folder_path)
    plot_loss_with_time(seq_dict, param_dict, dx, in_images, mode='fft_magnitude', foldername=folder_path)
    plot_loss_with_time(seq_dict, param_dict, dx, in_images, mode='fft_phase', foldername=folder_path)
    plot_loss_with_time(seq_dict, param_dict, dx, in_images, mode='gradient', foldername=folder_path)
    plot_loss_with_time(seq_dict, param_dict, dx, in_images, mode='laplace', foldername=folder_path)

    # Physical quantities
    plot_physics(seq_dict, param_dict, dx, foldername=folder_path)

    # PDFs
    plot_pdf_single(seq_dict, foldername=folder_path)
    plot_pdf_images(seq_dict, param_dict, in_images, foldername=folder_path)

    # R^2 (not used)
    plot_rsquared(seq_dict, param_dict, in_images, foldername=folder_path)

    # POD spectrum plot of phi, n and omega
    plot_POD(seq_dict, param_dict, dx, in_images, foldername=folder_path)

    # Q criterion, distribution and snapshot of all fields
    plot_Q_criterion(seq_dict, dx, in_images, time_length=400, foldername=folder_path)