import optuna

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap

from functools import partial
from typing import Dict, Tuple

from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict

from utils.model_checkpointer import save_model
from utils.data_handling import JAXDataLoader, seq_to_X
from utils.trainstate_init import ExtendedTrainState
from utils.physical_quantities import periodic_gradient, periodic_laplace

# This file contains the training module used for the Neural Networks.


def train_model(
        key: jax.random.PRNGKey,
        state: ExtendedTrainState,
        train_loader: JAXDataLoader,
        val_loader: JAXDataLoader,
        epochs: int,
        in_images: int,
        alpha: float = 0.5,
        y_diff: bool = False,
        save_path: str = None,
        trial: optuna.Trial = None
) -> Tuple[ExtendedTrainState, Dict]:
    """
    Training loop for the Neural Networks.

    Args:
        key (jax.random.PRNGKey): JAX RNG key.
        state (ExtendedTrainState): model state.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validation data loader.
        epochs (int): number of epochs to train.
        in_images (int): number of input images for the model.
        resolution (int): resolution of the data.
        alpha (float): weight scalar for the auxiliary loss term.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.
        save_path (str): path to save the model.
        trial (optuna.Trial): optuna trial object in case of hyperparameter optimization.

    Returns:
        state (ExtendedTrainState): trained model state.
        metrics (dict): training metrics.

    """
    metrics = {'train_loss': [], 'val_loss': [], 'val_aux_loss': [], 'aux_loss': []}
    best_error = 1.0 # initialize with high value, to find best model

    # main loop
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Initialize loss lists
        train_batch_loss = []
        aux_batch_loss = []

        # training loop
        for sequence_batch in train_loader:
            subkey, key = jax.random.split(key) 
            state, train_loss, aux_loss = curriculum_train_step(
                                            state, 
                                            subkey,
                                            sequence_batch, 
                                            epoch=epoch, 
                                            max_epochs=epochs,
                                            in_images=in_images,
                                            alpha=alpha,
                                            y_diff=y_diff)
            # Report losses
            train_batch_loss.append(train_loss)
            aux_batch_loss.append(aux_loss)
        metrics['train_loss'].append(jnp.mean(jnp.stack(train_batch_loss)))
        metrics['aux_loss'].append(jnp.mean(jnp.stack(aux_batch_loss)))

        # validation loop
        val_batch_loss = []
        val_aux_loss = []
        for sequence_batch in val_loader:
            val_loss, aux_loss = curriculum_val_step(state, 
                                           sequence_batch, 
                                           in_images, 
                                           y_diff=y_diff)
            val_batch_loss.append(val_loss)
            val_aux_loss.append(aux_loss)
        metrics['val_loss'].append(jnp.mean(jnp.stack(val_batch_loss)))
        metrics['val_aux_loss'].append(jnp.mean(jnp.stack(val_aux_loss)))

        # Print current losses
        print('CURRENT SINGLESTEP LOSS: ', metrics['val_loss'][-1])
        print('CURRENT MULTISTEP LOSS: ', metrics['val_aux_loss'][-1])

        # Save the model if it is the best one found so far. Only save after 20% of epochs
        current_error = metrics['val_loss'][-1]
        if (current_error < best_error) and epoch > (epochs / 5) and save_path:
            best_error = current_error
            save_model(state, save_path, 'best_model', config=state.config)
            print('New best model found at epoch: ', epoch)
        
        # Report the loss to optuna if hyperparameter optimization is used, and prune if necessary
        if trial:
            trial.report(metrics['val_loss'][-1], step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    # Save the final model
    if save_path:
        save_model(state, save_path, f'final_model_epoch{epochs}', config=state.config)   
    print(50*'-')
    print('TRAINING COMPLETE')
    print(50*'-')    

    return state, metrics


@partial(jit, static_argnames=['in_images', 'y_diff'])
def curriculum_train_step(state, key, sequence, epoch, max_epochs, in_images, alpha, y_diff): 
    """
    Perform training step for curriculum schedule, vectorized over the batch dimension.

    Args:
        state (ExtendedTrainState): model state.
        key (jax.random.PRNGKey): random key for difficulty factor RNG.
        sequence (jnp.ndarray): training sample sequence of shape (x, y, t, features).
        epoch (int): current epoch.
        max_epochs (int): maximum number of epochs for the training.
        in_images (int): number of input images for the model.
        alpha (float): weight for the auxiliary loss.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        state (ExtendedTrainState): updated model state.
        train_loss (jnp.ndarray): training loss, according to the curriculum schedule.
        auxiliary_loss (jnp.ndarray): auxiliary loss for reference.

    """
    def batch_loss(params):
        #NOTE: We have to vectorize keys as well, as we need different RNG for each sample in the batch
        loss, auxiliary_loss, updates = vmap(curriculum_step, 
                    in_axes=(0, 0, None, None, None, None, None, None),
                    out_axes=(0, 0, None), 
                    axis_name='batch')(sequence, keys, params, state, 
                                       epoch, max_epochs, in_images, y_diff)
        # Save auxiliary loss in updates, as value and grad can only return one auxiliary output
        updates['auxiliary_loss'] = jnp.mean(auxiliary_loss)

        # total loss
        loss = jnp.mean(loss + alpha * auxiliary_loss)

        return loss, updates
    
    # Perform backpropagation
    keys = jax.random.split(key, sequence.shape[0])
    (loss, updates), grad = value_and_grad(batch_loss, has_aux=True)(state.params)

    # Clip gradients to avoid exploding gradients
    clip_val = clip_exp_decay(epoch, max_epochs)
    grad = clip_by_value(grad, -clip_val, clip_val)

    # Update model parameters
    state = state.apply_gradients(grads=grad)
    state = state.replace(batch_stats=updates['batch_stats'])

    train_loss = loss - alpha * updates['auxiliary_loss'] # report only basic loss

    return state, train_loss, updates['auxiliary_loss']


def curriculum_step(
        sequence: jnp.ndarray, 
        key: jax.random.PRNGKey, 
        params: FrozenDict, 
        state: ExtendedTrainState, 
        epoch: int, 
        max_epochs: int, 
        in_images: int, 
        y_diff: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Curriculum learning step for the NN model.

    Args:
        sequence (jnp.ndarray): training sample sequence of shape (x, y, t, features).
        key (jax.random.PRNGKey): random key for difficulty factor RNG.
        params (FrozenDict): model parameters.
        state (ExtendedTrainState): model state.
        epoch (int): current epoch.
        max_epochs (int): maximum number of epochs for the training.
        in_images (int): number of input images for the model.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        loss (jnp.ndarray): L2 loss between the prediction using TF and the ground truth sequence.
        auxiliary_loss (jnp.ndarray): auxiliary loss based on the gradients and laplacians of the fields.
        updates (dict): updated batch statistics.

    """
    # initialize
    resolution = sequence.shape[0]
    variables = {'params': params, 'batch_stats': state.batch_stats}
    input_sequence = sequence.copy()
    predictions = jnp.zeros_like(sequence)
    predictions = predictions.at[:, :, :in_images, :].set(sequence[:, :, :in_images, :])

    # Set difficulty factor based on the epoch
    difficulty_factor = (epoch / max_epochs)

    # Roll through the sequence and fill the prediction array
    for t in range(in_images, sequence.shape[2]):
        # split key
        train_key, key = jax.random.split(key)

        # Determine whether to use ground truth or model predictions based on difficulty_factor
        X_in = input_sequence[:, :, t - in_images:t, :]
        prob_array = jax.random.uniform(train_key, shape=(X_in.shape))
        X_array = jnp.where(prob_array < difficulty_factor, predictions[:, :, t - in_images:t, :], X_in)

        # Transform curriculum modified sequence to X
        X = seq_to_X(X_array, resolution)

        # Predict the next timestep, if y_diff is True, add prior timestep to recover snapshot
        pred, updates = state.apply_fn(variables, X, train=True, mutable=['batch_stats'])
        pred = jax.lax.cond(
            jnp.any(y_diff),
            lambda p: p + X_in[:, :, -1, :], 
            lambda p: p,
            pred
        )
        predictions = predictions.at[:, :, t, :].set(pred)

    # Compute loss
    loss, auxiliary_loss = l2_loss(predictions, sequence, resolution)

    return loss, auxiliary_loss, updates


@partial(jit, static_argnames=['in_images', 'y_diff']) # set static argnames for jit
def curriculum_val_step(
    state: ExtendedTrainState, 
    sequence: jnp.ndarray, 
    in_images: int, 
    y_diff: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform validation step for curriculum schedule, vectorized over the batch dimension.

    Args:
        state (ExtendedTrainState): model state.
        sequence (jnp.ndarray): training sample sequence of shape (x, y, t, features).
        in_images (int): number of input images for the model.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        val_loss (Tuple[jnp.ndarray, jnp.ndarray]): L2 loss of TF predictions and free running predictions.

    """
    def batch_loss(params): 
        loss, self_loss = vmap(validation_step, 
                    in_axes=(0, None, None, None, None), 
                    out_axes=0, 
                    axis_name='batch')(sequence, params, state, in_images, y_diff)
        return jnp.mean(loss), jnp.mean(self_loss)
    val_loss = batch_loss(state.params)

    return val_loss


def validation_step(
        sequence: jnp.ndarray, 
        params: FrozenDict, 
        state: ExtendedTrainState, 
        in_images: int, 
        y_diff: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Validation loss step for the NN model.

    Args:
        sequence (jnp.ndarray): training sample sequence of shape (x, y, t, features).
        params (FrozenDict): model parameters.
        state (ExtendedTrainState): model state.
        in_images (int): number of input images for the model.
        resolution (int): resolution of the sequence data.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        loss (jnp.ndarray): L2 loss between the prediction using TF and the ground truth sequence.
        loss_self (jnp.ndarray): L2 loss between the free running sequence and the ground truth sequence.

    """
    # Initialize variables + the predictions with the input sequence
    predictions = jnp.zeros_like(sequence)
    predictions = predictions.at[:, :, :in_images, :].set(sequence[:, :, :in_images, :])
    self_predictions = predictions.copy()

    resolution = predictions.shape[0]
    variables = {'params': params, 'batch_stats': state.batch_stats}

    # Roll through the sequence and fill the prediction array
    for t in range(in_images, sequence.shape[2]):
        input = sequence[:, :, t - in_images:t, :].copy()
        input_self = self_predictions[:, :, t - in_images:t, :].copy()

        X = seq_to_X(input, resolution)
        X_self = seq_to_X(input_self, resolution)

        pred = state.apply_fn(variables, X, train=False)
        pred_self = state.apply_fn(variables, X_self, train=False)

        # In case of predicting the difference, add the last timestep of the input sequence 
        # to recover the prediction snapshot
        pred = jax.lax.cond(
            jnp.any(y_diff),  
            lambda p: p + input[:, :, -1, :],  
            lambda p: p,  
            pred  
        )
        pred_self = jax.lax.cond(
            jnp.any(y_diff), 
            lambda p: p + input_self[:, :, -1, :], 
            lambda p: p, 
            pred_self  
        )
        # Set the predictions
        predictions = predictions.at[:, :, t, :].set(pred)
        self_predictions = self_predictions.at[:, :, t, :].set(pred_self)
    
    # Compute the L2 loss between the predictions and the ground truth sequence
    loss, _ = l2_loss(predictions, sequence, resolution)
    loss_self, _ = l2_loss(self_predictions, sequence, resolution)
    return loss, loss_self


def clip_exp_decay(
        epoch: int, 
        max_epochs: int, 
        init_value: float=10.0, 
        final_value: float=0.001, 
        transition_begin: int=0
) -> float:
    """
    Exponential decay of the clipping value for the gradients.
    By default starts at 10, quickly drops to ~1 after 10 epochs, 0.5 after 20 etc.
    optimised for 100 epochs of training.

    """
    rate_factor = (epoch - transition_begin) / max_epochs
    clip_value = init_value * (final_value ** (rate_factor ** 0.5))

    return clip_value


def clip_by_value(
        grads: Dict, 
        min_value: float, 
        max_value: float
) -> Dict:
    """
    Clip gradients to a specified range.

    Args:
        grads (Dict): model gradients.
        min_value (float): minimum value for the gradients.
        max_value (float): maximum value for the gradients.

    Returns:
        clipped_grads (Dict): clipped gradients.
    
    """

    return jax.tree_map(lambda g: jnp.clip(g, min_value, max_value), grads)


def l2_loss(prediction: jnp.ndarray, 
            sequence: jnp.ndarray, 
            resolution: int=128
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the relative L2 loss between the prediction and the ground truth sequence.
    Additionally compute the auxiliary loss based on the gradients and laplacians of the fields.

    Args:
        prediction (jnp.ndarray): Predicted sequence of shape (x, y, t, features).
        sequence (jnp.ndarray): Ground truth sequence of shape (x, y, t, features).
        resolution (int): Resolution of the sequence data.

    Returns:
        loss (jnp.ndarray): L2 loss between the prediction and the ground truth sequence.
        auxiliary_loss (jnp.ndarray): Auxiliary loss based on the gradients and laplacians of the fields.

    """
    # Assume k0 = 0.15 to obtain dx
    dx = 2* jnp.pi / 0.15 / resolution

    # Transpose the sequences to (feature, time, x, y) format for the gradient functions
    transposed_sequence = sequence.transpose(3, 2, 0, 1)
    transposed_prediction = prediction.transpose(3, 2, 0, 1)

    # Get gradients and laplacians of the fields
    grad_seqy = periodic_gradient(transposed_sequence, dx, axis=-1)
    grad_seqx = periodic_gradient(transposed_sequence, dx, axis=-2)
    grad_seq = grad_seqx + grad_seqy
    laplace_seq = periodic_laplace(transposed_sequence, dx)  

    grad_predy = periodic_gradient(transposed_prediction, dx, axis=-1)
    grad_predx = periodic_gradient(transposed_prediction, dx, axis=-2)
    grad_pred = grad_predx + grad_predy
    laplace_pred = periodic_laplace(transposed_prediction, dx) 

    # Compute the relative L2 loss
    grad_loss = jnp.linalg.norm(grad_seq - grad_pred) / jnp.linalg.norm(grad_seq)
    laplace_loss = jnp.linalg.norm(laplace_seq - laplace_pred) / jnp.linalg.norm(laplace_seq)
    auxiliary_loss = (grad_loss + laplace_loss)
    loss = jnp.linalg.norm(prediction - sequence) / jnp.linalg.norm(sequence)

    return loss, auxiliary_loss


def run_sequence(
        sequence: jnp.ndarray, 
        state: ExtendedTrainState, 
        rms: jnp.ndarray, 
        in_images: int=5, 
        resolution: int=128, 
        y_diff: bool=False
)-> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run a sequence through the model from posterior testing.

    Args:
        sequence (jnp.ndarray): sequence of shape (x, y, t, features).
        state (ExtendedTrainState): model state.
        rms (jnp.ndarray): RMS values for the phi and density fields.
        in_images (int): number of input images for the model.
        resolution (int): resolution of the data.
        y_diff (bool): whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        gt_sequence (jnp.ndarray): ground truth sequence.
        norm_prediction (jnp.ndarray): normalized prediction sequence.

    """
    # Renormalize the actual sequence
    gt_sequence = sequence * rms

    # Initialize the normalized and non-normalized prediction arrays
    prediction = jnp.zeros_like(sequence)
    norm_prediction = jnp.zeros_like(sequence)
    prediction = prediction.at[:, :, :in_images, :].set(sequence[:, :, :in_images, :])
    norm_prediction = norm_prediction.at[:, :, :in_images, :].set(gt_sequence[:, :, :in_images, :])

    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    # Roll through the sequence and fill the prediction array
    for t in tqdm(range(in_images, sequence.shape[2]), desc='Running Sequence'):
        input = prediction[:, :, t - in_images:t, :].copy()
        X = seq_to_X(input, resolution)
        pred = state.apply_fn(variables, X, train=False)

        # In case of predicting the difference, add the last timestep of the input sequence
        pred = jax.lax.cond(
            jnp.any(y_diff),
            lambda p: p + input[:, :, -1, :],
            lambda p: p, 
            pred 
        )

        # Set the predictions
        prediction = prediction.at[:, :, t, :].set(pred)
        norm_prediction = norm_prediction.at[:, :, t, :].set(pred * rms)

    return gt_sequence, norm_prediction