import os
import shutil

import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.core.frozen_dict import FrozenDict
import optax

from modules.FNO_modules import UFNO2D, FNO2D
from modules.ResNet import ResNet
from modules.Unet import Unet

from utils.trainstate_init import ExtendedTrainState

# This file contains functions to save and load model checkpoints.

def save_model(
        state: ExtendedTrainState, 
        save_path: str, 
        save_name: str, 
        config: FrozenDict
) -> None:
    """
    A simple function to save the model's state and configuration to a checkpoint directory.

    Args:
        state: Current TrainState to be saved.
        save_path: Base path where the checkpoint should be saved.
        save_name: Name of the checkpoint directory - e.g. epoch number or best_model.
        config: Configuration dictionary to inituate the model.
    
    """
    # Access and create the checkpoint directory - overwrite if it exists
    ckptdir = os.path.abspath(save_path + save_name)
    if os.path.exists(ckptdir):
       shutil.rmtree(ckptdir)
    os.makedirs(ckptdir, exist_ok=True)

    # Initialize the checkpointer and checkpoint
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)

    # Save the checkpoint
    orbax_checkpointer.save(ckptdir, ckpt, save_args, force=True)

def load_model(
        load_path: str, 
        save_name: str, 
        learning_rate: float = None,
        decay_rate: float = None,
        num_train_steps: int = None
) -> ExtendedTrainState:
    """
    A simple function to load the model's state and configuration from a checkpoint directory,
    as saved by the save_model function.

    Args:
        load_path (str): Base path where the checkpoint should be loaded from.
        save_name (str): Name of the checkpoint directory - e.g. epoch number or best_model.
        learning_rate (float): Learning rate for the optimizer.
        decay_rate (float): Decay rate of the learning rate.
        num_train_steps (int): Number of training steps for learning rate scheduling.

    Returns:
        restored_state (ExtendedTrainState): The restored TrainState.

    """
    ckptdir = os.path.abspath(load_path + save_name)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(ckptdir)
    restored_state = ckpt['model']

    # remove 'key' from config, in case it was saved
    if isinstance(ckpt['config'], FrozenDict):
        config = ckpt['config'].unfreeze()
    else:
        config = ckpt['config']
    config.pop('key', None)

    # Select right model based on the path label and initialize it
    load_path = load_path.lower().rstrip('/')
    if load_path.endswith('ufno'):
        model = UFNO2D(**ckpt['config'])
    elif load_path.endswith('fno'):
        model = FNO2D(**ckpt['config'])
    elif load_path.endswith('resnet'):
        model = ResNet(**ckpt['config'])
    elif load_path.endswith('unet'):
        model = Unet(**ckpt['config'])
    
    # Initialize the optimizer and learning rate schedule in case of training continuation
    if learning_rate:
        lr_schedule = optax.exponential_decay(learning_rate, 
                                          transition_steps=num_train_steps, 
                                          decay_rate=decay_rate)
    else:
        lr_schedule = 1e-3 # default learning rate

    # Restore the TrainState
    restored_state = ExtendedTrainState(
        apply_fn=model.apply,
        config=ckpt['config'],
        params=restored_state['params'],
        batch_stats=restored_state['batch_stats'],
        tx=optax.adam(learning_rate=lr_schedule),
        opt_state=restored_state['opt_state'],
        step=restored_state['step']
    )

    return restored_state


def get_model_type(model_path: str):
    """
    Get the type of the model based on the model path. Assume that the model path ends with the model type.

    Args:
        model_path (str): Path to the model

    """
    model_path = model_path.lower().rstrip('/')  # Remove trailing slashes

    if model_path.endswith('ufno'):
        model = 'U-FNO'
    elif model_path.endswith('fno'):
        model = 'FNO'
    elif model_path.endswith('resnet'):
        model = 'ResNet'
    elif model_path.endswith('unet'):
        model = 'U-Net'
    else:
        raise ValueError("Model not recognised.")
    return model