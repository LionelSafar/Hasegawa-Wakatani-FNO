import os
import argparse

import optuna

from jax import random
from flax.core.frozen_dict import FrozenDict

from modules.ResNet import ResNet
from modules.Unet import Unet
from modules.FNO_modules import FNO2D, UFNO2D

from utils.trainstate_init import initialize_trainstate
from utils.data_handling import SequenceDataset, JAXDataLoader, train_test_split, get_sample

from model_training.training_modules import train_model

# This script is used to optimise hyperparameters for the FNO model using Optuna.
# The hyperparameter ranges can be adjusted in the objective function.

if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='fno', 
                           help='Model to optimise, fno, resnet or unet'
                           )
    argparser.add_argument('--data_path', type=str, default='data', 
                           help='Path to data to use'
                           )
    argparser.add_argument('--images', type=int, default=None, 
                           help='Number of input images, if None - optimise'
                           )
    argparser.add_argument('--epochs', type=int, default=20, 
                           help='Number of epochs'
                           )
    argparser.add_argument('--n_trials', type=int, default=25, 
                           help='Number of trials'
                           )
    argparser.add_argument('--name', type=str, default='fno_resnet', 
                           help='Name of study'
                           )
    argparser.add_argument('--resolution', type=int, default=64, 
                           help='Resolution of data'
                           )
    args = argparser.parse_args()
    model = args.model
    data_path = args.data_path
    images = args.images
    epochs = args.epochs
    n_trials = args.n_trials
    name = args.name
    resolution = args.resolution

    # Set up random key
    key = random.PRNGKey(0)
    split_key, model_key, init_key, shuffle_key, sample_key = random.split(key, num=5)

    # Load data
    dataset = SequenceDataset(data_path)
    train_dataset, val_dataset = train_test_split(split_key, dataset, 0.8)
    out_channels = 2 

    def objective(trial: optuna.Trial):
        """
        
        Objective function to optimise hyperparameters for the model.

        Args:
            trial (optuna.Trial): Optuna trial object.

        """
        if images is None: # optimize number of input images (optional)
            in_images = trial.suggest_int('in_images', 4, 10)
        else:
            in_images = images
        

        X_sample, _ = get_sample(sample_key, train_dataset, in_images=in_images)
        in_channels = 2 + 2 * in_images
        key = random.PRNGKey(0)
        train_key, loader_key = random.split(key)

        # Define model, set hyperparamter range to search
        if model.lower() == 'fno':
            mode = trial.suggest_int('modes', 16, 28)
            config = FrozenDict({'in_channels': in_channels,
                        'out_channels': out_channels,
                        'modes': (mode, mode),
                        'hidden_channels': trial.suggest_int('hidden_channels', 40, 80),
                        'n_layers': 4})
            fno = FNO2D(**config)
        elif model.lower() == 'resnet':
            hc = trial.suggest_int('hidden_channels', 8, 32)
            n_blocks = trial.suggest_int('num_blocks', 3, 6)
            config = FrozenDict({'out_channels': out_channels,
                        'hidden_channels': (hc, hc, hc),
                        'num_blocks': (n_blocks, n_blocks, n_blocks)})
            fno = ResNet(**config)
        elif model.lower() == 'unet':
            hc = trial.suggest_int('hidden_channels', 18, 36)
            depth = trial.suggest_int('depth', 2, 4)
            config = FrozenDict({'initial_hidden_channels': hc,
                    'out_channels': out_channels,
                    'depth': depth})
            fno = Unet(**config)
        elif model.lower() == 'ufno':
            mode = trial.suggest_int('modes', 6, 14)
            config = FrozenDict({'in_channels': in_channels,
                        'out_channels': out_channels,
                        'modes': (mode, mode),
                        'hidden_channels': trial.suggest_int('hidden_channels', 14, 28),
                        'n_layers': 4})
            fno = UFNO2D(**config)
        
        # Additional hyperparameters: batch size, learning rate, decay factor
        batch_size = trial.suggest_int('batch_size', 8, 24)
        lr = trial.suggest_float('learning_rate', 1e-3, 2e-1, log=True)
        decay_rate = trial.suggest_float('decay_factor', 9e-3, 9e-1, log=True)
        num_train_steps = int(len(train_dataset) * epochs // batch_size)

        # Initialize training state and data loader
        state = initialize_trainstate(init_key, fno, X_sample, config, 
                                    num_train_steps=num_train_steps, learning_rate=lr,
                                    decay_rate=decay_rate)
        train_loader = JAXDataLoader(loader_key, train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = JAXDataLoader(loader_key, val_dataset, batch_size=batch_size, shuffle=False)

        # Train model, allowing Optuna to prune bad trials
        state, metrics = train_model(train_key, state, train_loader, val_loader, epochs,
                                    in_images, resolution=resolution, alpha=0.8, trial=trial)
        
        return metrics['val_loss'][-1]

    # Create study and optimise
    study = optuna.create_study(direction='minimize',
                                study_name=f'{name}',
                                storage=f'sqlite:///fno_{name}.db',
                                pruner=optuna.pruners.MedianPruner(),
                                load_if_exists=True)
    study.optimize(objective, n_trials=n_trials-len(study.trials), show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print(f'Best Validation Accuracy: {trial.value:4.2}')
    print(f'Best Params:')
    for key, value in trial.params.items():
        print(f'-> {key}: {value}')
    print('Best trial number:', trial.number)

    print('All trials:', study.trials)
    print('All trial values:', [trial.value for trial in study.trials])
    bad_trials = [trial for trial in study.trials if trial.value is None]
    good_trials = [trial for trial in study.trials if trial.value is not None and trial.value < 0.3]
    print('Number of good trials:', len(good_trials))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', 'figures')

    # Visualise the optimisation

    # Save parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(path + f'/param_importance_{name}.pdf', scale=3)

    # Save parallel coordinate
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    fig2.update_layout(
        title_text='',  # Remove title
        font=dict(
            size=16  # Increase label and tick size
        )
    )
    fig2.write_image(path + f'/parallel_coordinate_{name}.pdf', scale=3)

    # Visualise slice
    fig3 = optuna.visualization.plot_slice(study)
    fig3.write_image(path + f'/slice_{name}.pdf', scale=3)