import os
import argparse
import shutil

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from flax.core.frozen_dict import FrozenDict

from modules.ResNet import ResNet
from modules.Unet import Unet
from modules.FNO_modules import FNO2D, UFNO2D

from model_training.training_modules import train_model

from utils.physical_quantities import get_physics
from utils.data_handling import SequenceDataset, JAXDataLoader, train_test_split, get_sample, preprocessing
from utils.trainstate_init import initialize_trainstate
from utils.model_checkpointer import load_model
from utils.run_sequence import run_sequence

from visualization.training_plotting_tools import *

if __name__ == "__main__":
    # Parsing
    parser = argparse.ArgumentParser(description="Train a NN model to learn HW with specified parameters.")
    parser.add_argument("--data_path", type=str, default="/scratch/izar/safar/c1_nu1e-09_512_0.02", 
                        help="data path that contains all HW simulation data to be used")
    parser.add_argument("--k", type=float, default=1.0, 
                        help="Background density gradient length used in the HW simulation")
    parser.add_argument("--c1", type=int, default=1, 
                        help="Adiabaticity parameter used in HW simulation")
    parser.add_argument("--nu", type=float, default=1e-09, 
                        help="Viscosity used in HW simulation")
    parser.add_argument("--k0", type=float, default=0.15, 
                        help="characteristic wavenumber of the slab")
    parser.add_argument("--simulation_timestep", type=float, default=0.02, 
                        help="simulation timestep in stored HW simulation")
    parser.add_argument("--simulation_resolution", type=int, default=256, 
                        help="image Resolution in stored simulation data")
    parser.add_argument("--hyperdiff", type=int, default=3,
                        help="Hyperdiffusion parameter used in HW simulation")
    parser.add_argument("--model", type=str, default='True', 
                        help="model choice: options are FNO, UFNO, ResNet, Unet")
    parser.add_argument("--diff_y", type=str, default='False', 
                        help="predict difference of consecutive images instead of the next image")
    parser.add_argument("--NN_timesteps", type=int, default=100, 
                        help="simulation timestep per NN timestep")
    parser.add_argument("--in_images", type=int, default=5, 
                        help="Number of input images for the model to predict next timestep")
    parser.add_argument("--testing_files", type=int, default=1, 
                        help="Number of files to use for a posteriori testing, will be subtracted from N_files")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--truncate_mode", type=int, default=16, 
                        help="max kx, ky modes to keep in case of FNO")
    parser.add_argument("--t_start", type=int, default=200, 
                        help="Start time of the simulation")
    parser.add_argument("--resolution", type=int, default=64, 
                        help="Resolution for the model to use: 32, 64, 128, 256")
    parser.add_argument("--N_files", type=int, default=1,
                        help="Number of files to use in total")
    parser.add_argument("--sequence_length", type=int, default=None,
                        help="Number of time steps per reduced sequence - If None use whole sequence")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for training")
    parser.add_argument("--best_model", type=str, default='False',
                        help="Use the best model instead of the final model")
    parser.add_argument("--snapshot", type=int, default=1,
                        help="time downsampling used to store the simulation, default is 1")
    parser.add_argument("--fraction", type=float, default=0.8,
                        help="Fraction of the timesequence to use for training")
    parser.add_argument("--hidden_channels", type=int, default=40,
                        help="Number of hidden channels for the model for FNO or initial hidden channels for others")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of model -- n_fourier layer for FNO, n_blocks for ResNet, n_layers for Unet")
    parser.add_argument("--id_run", type=str, default='',
                        help="Identifier for the run for saving purposes (Optional)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the model")
    parser.add_argument("--decay_rate", type=float, default=0.1,
                        help="Decay rate for the learning rate")
    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments as variables
    data_path = args.data_path
    k = args.k
    c1 = args.c1
    nu = args.nu
    k0 = args.k0
    simulation_timestep = args.simulation_timestep
    simulation_resolution = args.simulation_resolution
    hyperdiff = args.hyperdiff
    model = args.model.lower()
    testing_files = args.testing_files
    diff_y = args.diff_y == 'True'
    NN_timesteps = args.NN_timesteps
    in_images = args.in_images
    epochs = args.epochs
    truncate_mode = args.truncate_mode
    t_start = args.t_start
    N_files = args.N_files
    resolution = args.resolution
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    best_model = args.best_model == 'True'
    snapshot = args.snapshot
    fraction = args.fraction
    hidden_channels = args.hidden_channels
    depth = args.depth
    id_run = args.id_run
    lr = args.learning_rate
    decay_rate = args.decay_rate

    # Constants
    downsampling = simulation_resolution // resolution # downsampling factor
    if downsampling not in [1, 2, 4, 8]:
        raise ValueError('Choose a valid resolution: 32, 64, 128, 256')
    L = 2 * jnp.pi / k0
    dx = L / resolution # grid spacing
    dt = simulation_timestep * NN_timesteps * snapshot # physical time step
    N = N_files - testing_files # number of files for training

    # Create labels for settings to save
    if diff_y:
        diff = 'diffy_'
    else:
        diff = ''
    if sequence_length:
        seq = f'_seq{sequence_length}'
    else:
        seq = '_fullseq'
    if model == 'fno':
        hp = f'_mod{truncate_mode}'
    elif model == 'resnet':
        hp = f''
    elif model == 'unet':
        hp = f''
    elif model == 'ufno':
        hp = f'_mod{truncate_mode}'

    # Create settings strings
    settings = f'k{k}_c1{c1}_nu{nu}_{resolution}res_st{t_start}+end_{NN_timesteps}t_{N_files}N_{diff}{seq}{model}'
    simulation_settings = f'k{k}_c1{c1}_nu{nu}'
    settings = f'{resolution}res_st{t_start}_{NN_timesteps}t_{N_files}N_{in_images}nimg_{diff}{seq}{id_run}{model}'
    dataset_settings = f'{resolution}res_st{t_start}_{NN_timesteps}t_{N_files}N_{seq}'

    # Create paths for saving
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, '..', 'Data/'))
    folder_path = os.path.join(path, f'{simulation_settings}/')
    save_path = os.path.join(folder_path, 'models/')
    dataset_path = os.path.join(folder_path, 'datasets/')
    figure_path = os.path.join(folder_path, 'figures/')
    processed_data_path = os.path.join(dataset_path, f'{dataset_settings}/')
    os.makedirs(path, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    print('figurepath', figure_path)

    # Create copy folder for backup data on the same level as the data folder (Generally on scratch)
    #copy_datapath = os.path.abspath(os.path.join(data_path, '', 'copy/', f'{dataset_settings}/', f'dataset_settings/'))
    #os.makedirs(os.path.abspath(os.path.join(data_path, '', 'copy/')), exist_ok=True)
    #os.makedirs(os.path.abspath(os.path.join(data_path, '', 'copy/', f'{dataset_settings}/')), exist_ok=True)
    #os.makedirs(copy_datapath, exist_ok=True)

    save_path = os.path.join(save_path, f'{settings}/')
    figure_path = os.path.join(figure_path, f'{settings}/')
    os.makedirs(figure_path, exist_ok=True)

    # Print settings of the run
    print(50*'-')
    print('SELECTED SIMULATION SETTINGS:', simulation_settings)
    print('SELECTED MODEL SETTINGS:', settings)

    # Take the processed data from the backup folder - slower but allows to directly run from /scratch
    #if copy_file:
    #    processed_data_path = copy_datapath

    # If no processed data is found in indicated folder, preprocess the data
    if not os.path.exists(processed_data_path):
        print(50*'-')
        print('NO PREPROCESSED DATA FOR CHOSEN SETTINGS FOUND: PREPROCESSING DATA')
        print(50*'-')
        rms = preprocessing(
                data_path = data_path, 
                out_path = processed_data_path,
                train_test_split = (N, testing_files),
                resolution=resolution,
                t_start = t_start,
                simulation_timestep = simulation_timestep * snapshot,
                NN_timestep = NN_timesteps,
                sequence_length = sequence_length
                )
        print('DATA PREPROCESSED SUCCESSFULLY')
        print(50*'-')

    else:
        print(50*'-')
        print('PREPROCESSED DATA FOUND FOR CHOSEN SETTINGS')
        rms = jnp.load(os.path.join(processed_data_path, 'rms_train.npy'))
        print(50*'-')

    # Copy datasets to the backup folder
    #if not copy_file:
    #    for file_name in os.listdir(processed_data_path):
    #        full_file_name = os.path.join(processed_data_path, file_name)
    #        if os.path.isfile(full_file_name):  # Ensure it's a file
    #            shutil.copy(full_file_name, copy_datapath)

    # Key initialisation
    key = jax.random.PRNGKey(0)
    dataset_key, dataloader_key, sample_key, model_key, init_key, train_key, subkey = jax.random.split(key, num=7)

    # Initialise dataset and dataloader
    train_datapath = os.path.join(processed_data_path, 'training_data.h5')
    test_datapath = os.path.join(processed_data_path, 'post_test_data.h5')

    dataset = SequenceDataset(train_datapath)
    test_dataset = SequenceDataset(test_datapath)

    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, key=dataset_key)

    train_loader = JAXDataLoader(dataloader_key, train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = JAXDataLoader(dataloader_key, val_dataset, batch_size=batch_size, shuffle=False)

    num_train_steps = int(len(train_dataset) * epochs // batch_size)

    # Print dataset lengths
    print(50*'-')
    print('TRAIN-DATASET LENGTH', len(train_dataset))
    print('VAL-DATASET LENGTH', len(val_dataset))
    print('TEST-DATASET LENGTH', len(test_dataset))
    print(50*'-')

    # Input and output channel dimension
    in_channels = 2 + 2 * in_images # 2 for grid, 2 times in_images for phi, density
    out_channels = 2 # phi, density

    # Select model
    if model == 'fno':
        config = FrozenDict(
                {'in_channels': in_channels, 
                 'out_channels': out_channels, 
                 'modes': (truncate_mode, truncate_mode), 
                 'hidden_channels': hidden_channels,
                 'n_layers': depth}
                 )
        model = FNO2D(**config)
    elif model == 'resnet':
        config = FrozenDict(
                {'out_channels': out_channels,
                 'hidden_channels': (hidden_channels, 2*hidden_channels, 4*hidden_channels),
                 'num_blocks': (depth, depth, depth)}
                 )
        model = ResNet(**config)
    elif model == 'unet':
        config = FrozenDict(
                {'initial_hidden_channels': hidden_channels,
                 'out_channels': out_channels,
                 'depth': depth})
        model = Unet(**config)
    elif model == 'ufno':
        config = FrozenDict(
                {'in_channels': in_channels,
                 'out_channels': out_channels,
                 'modes': (truncate_mode, truncate_mode),
                 'hidden_channels': hidden_channels,
                 'n_layers': depth}
                 )
        model = UFNO2D(**config)

    # Get a random sample from the validation dataset to initialize
    X_sample, y_sample = get_sample(
        sample_key, val_dataset, in_images=in_images, 
        resolution=resolution, y_diff=diff_y
        )

    state = initialize_trainstate(
        init_key, model, X_sample, config, num_train_steps=num_train_steps, 
        learning_rate=lr, decay_rate=decay_rate
        )

    # Print parameter count of the model
    flat_params, _ = tree_flatten(state.params)
    total_params = sum(jnp.size(param) for param in flat_params)
    print(f"{model} \n PARAMETER COUNT:", total_params)

    # Train the model
    state, metrics = train_model(
        train_key, state, train_loader, val_loader, epochs=epochs, 
        in_images=in_images, alpha=0.8,
        y_diff=diff_y, save_path=save_path
        )
    
    # Plot losses
    plot_train_val_losses(metrics, foldername=figure_path)

    # Some immediate testing for further model evaluation
    print(50*'-')
    print('TESTING MODEL PERFORMANCE')

    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    if best_model:
        state = load_model(save_path, 'best_model')
        print('Loaded best model')
    else:
        print('Loaded final model')

    # Run a training sequence to full scope and plot
    train_sequence = get_sample(subkey, train_dataset, sequential=True, resolution=resolution)
    gt_sequence, pred_sequence = run_sequence(
        train_sequence, state, rms, in_images, resolution, y_diff=diff_y
        )
    gt_stats, pred_stats = get_physics(
        pred_sequence, gt_sequence, dt=dt, dx=dx, 
        c1=c1, nu=nu, kappa=k, hyperdiff=hyperdiff, 
        inc_vorticity=False
        )
    plot_physics(gt_stats, pred_stats, foldername=figure_path, id='train')

    # Plot loss with time per variable
    plot_loss_with_time(gt_sequence, pred_sequence, dt=dt, inc_vorticity=False, 
                        foldername=figure_path, trainflag=fraction, id='train')

    # Testing sequence plots:
    test_sequence = get_sample(
        subkey, test_dataset, sequential=True, resolution=resolution
        )
    gt_test, pred_test = run_sequence(
        test_sequence, state, rms, in_images, resolution, y_diff=diff_y
        )
    gt_test_stats, pred_test_stats = get_physics(
        pred_test, gt_test, dt=dt, dx=dx, 
        c1=c1, nu=nu, kappa=k, hyperdiff=hyperdiff, 
        inc_vorticity=False
        )
    plot_physics(gt_test_stats, pred_test_stats, foldername=figure_path)
    plot_loss_with_time(gt_test, pred_test, dt=dt, inc_vorticity=False, foldername=figure_path)

    # Plot snapshots of the testing sequence
    plot_snapshot(gt_test, pred_test, variable='phi', n_images=8, dt=dt, foldername=figure_path)
    plot_snapshot(gt_test, pred_test, variable='density', n_images=8, dt=dt, foldername=figure_path)

    plot_snaps_and_grad(gt_test, pred_test, 'phi', n_images= 6, dt=dt, foldername=figure_path)
    plot_snaps_and_grad(gt_test, pred_test, 'density', n_images= 6, dt=dt, foldername=figure_path)