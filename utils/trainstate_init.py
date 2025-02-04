import jax
import jax.numpy as jnp

import flax
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict

import optax

from typing import Dict, List

# This file contains utility functions for the FNO model:
# - Initialization of the TrainState and the TrainState class
# - Running a sequence through the model for posterior testing


class ExtendedTrainState(TrainState):
    """
    Extended TrainState class for the NN models, that includes the configuration dictionary and
    the batch statistics.
    
    """
    config: dict 
    batch_stats: dict

    @classmethod
    def create(cls, apply_fn, 
               params: FrozenDict, 
               batch_stats: FrozenDict,
               tx: optax.GradientTransformation, 
               config: FrozenDict, 
               step: int = 0):
        
        return cls(
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            opt_state=tx.init(params),
            config=config,
            step=step
        )
    

def initialize_trainstate(
        init_key: jax.random.PRNGKey, 
        model: flax.linen.Module, 
        X: jnp.ndarray, 
        config: Dict, 
        num_train_steps: int, 
        learning_rate: float=5e-2, 
        decay_rate: float=0.1
) -> ExtendedTrainState:
    """
    Initialise the TrainState for the model, using the Adam optimizer and 
    exponential decay learning rate schedule.

    Args:
        init_key (jax.random.PRNGKey): key for initialization
        model (flax.linen.Module): Neural Network model
        X (jnp.ndarray): input data for the model of shape (x, y, channels) to initialise parameter shapes
        config (dict): configuration dictionary for the model
        num_train_steps (int): number of training steps for learning rate scheduling
        learning_rate (float): initial learning rate for the optimizer
        decay_rate (float): decay rate to which fraction the learning rate should be reduced

    Returns:
        state (ExtendedTrainState): initial TrainState

    """
    # Initialize the model with the input data to get the parameter shapes
    # NOTE: Batch dimension is not needed here as we will vectorize over it later
    variables = model.init(init_key, X, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']

    # Define the learning rate schedule
    lr_schedule = optax.exponential_decay(
            learning_rate, 
            transition_steps=num_train_steps, 
            decay_rate=decay_rate)
    #lr_schedule2 = cosine_annealing_with_decay(learning_rate, T_0, T_mult, decay_factor, max_steps)
    optimizer = optax.adam(learning_rate=lr_schedule)
    
    # Create the TrainState
    state = ExtendedTrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optimizer,
        config=config
    )
    return state


def cosine_annealing_with_decay(
        base_lr: float, 
        T_0: int, 
        T_mult: int=2, 
        decay_factor: float=0.9, 
        max_steps: int=100
) -> List:
    """
    Cosine annealing with restarts and decayed maximum learning rate.
    NOTE: This function is not used currently, but was used to test a different learning rate schedule.
    
    Parameters:
        base_lr (float): Initial learning rate.
        T_0 (int): Number of iterations in the first cycle.
        T_mult (int): Multiplier for cycle length after each restart.
        decay_factor (float): Factor to decay the max learning rate after each restart.
        max_steps (int): Total number of steps.
    
    Returns:
        lr_values (list): Learning rate values for all steps.
    """
    lr_values = []
    current_base_lr = base_lr
    step = 0
    cycle_length = T_0

    while step < max_steps:
        for t in range(int(cycle_length)):
            if step >= max_steps:
                break
            # Cosine annealing within the current cycle
            lr = current_base_lr * (1 + jnp.cos(jnp.pi * t / cycle_length)) / 2
            lr_values.append(lr)
            step += 1
        # After each cycle, update the base learning rate and cycle length
        current_base_lr *= decay_factor
        cycle_length *= T_mult

    return lr_values
