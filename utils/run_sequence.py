import jax
import jax.numpy as jnp
from typing import Tuple
from tqdm import tqdm

from utils.trainstate_init import ExtendedTrainState
from model_training.training_modules import seq_to_X

# This file contains running function to roll the model through a sequence of snapshots and 
# predict a whole sequence.

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