import os
from typing import Iterator, List, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import scipy
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from functools import partial

# This file contains:
# - Functions for preprocessing simulation data for training the FNO model.
# - Dataset and DataLoader classes for loading the preprocessed data and associated helper functions.


def preprocessing(
        data_path: str, 
        out_path: str,
        train_test_split: Tuple[int, int] = (30, 1),
        resolution: int = 128,
        t_start: int = 0,
        simulation_timestep: float = 0.01,
        NN_timestep: int = 20,
        sequence_length: int = None,
        sample_size = None,
) -> jnp.ndarray:
    """
    Preprocess simulation data for training the NN models.

    Args:
        data_path (str): Path to the directory containing the simulation data.
        out_path (str): Path to the directory where the preprocessed data will be saved.
        train_test_split (tuple[int, int]): Number of files to use for training and testing.
        resolution (int): Resolution of the preprocessed data.
         -- If the resolution of the simulation data is higher, it will be downsampled
         -- to the specified resolution using an anti-aliasing filter.
        t_start (int): Start time t=cs/Ln of the simulation. Prior timesteps will be discarded.
        simulation_timestep (float): Timestep of the simulation.
        NN_timestep (int): Timestep of the FNO model.
        sequence_length (int): Length of the sequences to be used for each training sample.
        sample_size (int): Number of samples to save for training and testing (Optional).

    Returns:
        rms (jnp.ndarray): Root mean square values for the phi and density fields.

    """
    # Gather all .h5 files in the data directory
    file_paths = [os.path.join(data_path, file_name)
                           for file_name in os.listdir(data_path) if file_name.endswith(".h5")]
    if not file_paths:
        raise FileNotFoundError('No .h5 files found in the specified data directory')
    if sum(train_test_split) > len(file_paths):
        raise ValueError('The sum of the train and test sequences exceeds the number of files, '
                         'reduce the number of training or testing sequences or use more data files.')
    if sum(train_test_split) < 1:
        raise ValueError('Select at least one file for training and testing.')
    if not os.path.exists(out_path):
        print(out_path, 'not found, creating directory...')
        os.makedirs(out_path)

    # Split the files into training and testing sets
    train_files = file_paths[:train_test_split[0]]
    test_files = file_paths[-train_test_split[1]:]

    # Get the index of the start time
    start_index = int(t_start / simulation_timestep)

    # Create the training and testing directories in the output path
    train_out_path = os.path.join(out_path, 'training_data.h5')
    test_out_path = os.path.join(out_path, 'post_test_data.h5')
    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)

    # Store the file paths and output paths in dictionaries
    files = {'train': train_files, 'test': test_files}
    paths = {'train': train_out_path, 'test': test_out_path}

    for key in list(files.keys()): # iterate train and test files
        with h5py.File(paths[key], 'w') as f: # open the output file
            if key == 'train': # only save RMS values for training data
                rms = jnp.zeros((len(files[key]), 2))
            idx = 0 # index for sequence number that is being saved
            for i, file in tqdm(enumerate(files[key]), desc=f'Processing {key} Data'):
                with h5py.File(file, 'r') as f_in: # read the simulation data
                    phi = jnp.array(f_in['phi'][start_index::NN_timestep, 
                                        :, :]).transpose(1, 2, 0)
                    density = jnp.array(f_in['density'][start_index::NN_timestep, 
                                        :, :]).transpose(1, 2, 0)
                    
                    # Downsample the data to the specified resolution if necessary
                    if phi.shape[0] != resolution:
                        if phi.shape[0] < resolution:
                            raise ValueError('The resolution of the simulation data is '
                                             'lower than the specified resolution!')
                        phi = scipy.signal.resample(phi, num=resolution, axis=0)
                        phi = scipy.signal.resample(phi, num=resolution, axis=1)
                        density = scipy.signal.resample(density, num=resolution, axis=0)
                        density = scipy.signal.resample(density, num=resolution, axis=1)

                # Stack the phi and density fields
                sequence = jnp.stack([phi, density], axis=-1)

                # compute RMS values for phi and density fields
                if key == 'train':
                    rms = rms.at[i, 0].set(jnp.mean(jnp.square(phi), axis=(0, 1, 2)))
                    rms = rms.at[i, 1].set(jnp.mean(jnp.square(density), axis=(0, 1, 2)))

                if (sequence_length and key =='train'): # split into shorter sequences
                    sequence_list = split_into_shorter_sequences(sequence, sequence_length)
                    for sequence in sequence_list:
                        f.create_dataset(f'sequence{idx}', data=sequence, dtype=np.float32)
                        idx += 1
                        if idx == sample_size:
                            print(f'Sample size of {sample_size} reached!')
                            break
                else: # save the full sequence
                    f.create_dataset(f'sequence{idx}', data=sequence, dtype=np.float32)
                    idx += 1
                    if idx == sample_size:
                        print(f'Sample size of {sample_size} reached!')
                        break
                if idx == sample_size:
                    print(f'Only {idx} sequences saved!')
                    break
            if key == 'train': # save RMS values for training data
                rms = jnp.sqrt(jnp.mean(rms, axis=0))
                outfile = os.path.join(out_path, f'rms_{key}')
                jnp.save(outfile, rms)

    return rms


def split_into_shorter_sequences(
        array: jnp.ndarray, 
        sequence_length: int=60, 
        stride: int=5
) -> List[jnp.ndarray]:
    """
    Use a rolling window to split a longer sequence into shorter sequences of length sequence_length,
    with a stride of stride.

    Args:
        array (jnp.ndarray): Input array of shape (x, y, t, features).
        sequence_length (int): Length of each shorter sequence.

    Returns:
        sequence_list (List[jnp.ndarray]): where each element has shape (x, y, sequence_length, features).

    """
    # Get the total number of timesteps
    num_timesteps = array.shape[2]
    sequence_list = []
    for i in range(0, num_timesteps - sequence_length, stride):
        sequence = array[:, :, i:i+sequence_length, :]
        sequence_list.append(sequence)

    return sequence_list


class SequenceDataset(Dataset):
    """
    Modified PyTorch Dataset class for loading sequence data of shape (x,y,t,feature)

    """
    def __init__(self, 
                file_path: str, 
                fraction: float = None):
        """
        Load the sequence data from the specified .h5 file. 
        This class assumes that the data is stored in the format (x, y, t, features), and that
        next to the filepath, a file named 'rms_train.npy' is stored in the same directory, containing
        the RMS values for the phi and density fields.

        Args:
            file_path (str): Path to the .h5 file containing the sequence data.
            fraction (float): Fraction of the data to use for training.

        """
        # Load the RMS values for normalization
        self.file_path = file_path
        self.rms = jnp.load(os.path.abspath(os.path.join(file_path, '..', 'rms_train.npy')))
        self.fraction = fraction

        # Load data based on the fraction
        with h5py.File(self.file_path, 'r') as f:
            self.sequences = [key for key in f.keys()]
            if self.fraction is not None:
                N = f[self.sequences[0]][...].shape[2]

                t_train = int(self.fraction * N)
                self.data = {key: f[key][:, :, :t_train, :] for key in self.sequences}
            else:
                self.data = {key: f[key][:] for key in self.sequences}

    def __len__(self):
        """Returns the number of sequences in the dataset."""

        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Returns the normalized sequence data."""

        sequence = self.data[self.sequences[idx]]
        norm_sequence = sequence / self.rms
        return norm_sequence
    
    def get_rms(self):
        """Returns the RMS values for the sequences."""

        return self.rms
    

def train_test_split(
        dataset: SequenceDataset, 
        train_size: float=0.8,
        shuffle: bool=False,
        key: jax.random.PRNGKey=None, 
) -> Tuple[SequenceDataset, SequenceDataset]:
    """
    Split dataset into training and validation sets, with JAX RNG.

    Args:
        key (jax.random.PRNGKey): JAX RNG key.
        dataset (SequenceDataset): SequenceDataset object.
        train_size (float): Fraction of the dataset to use for training.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        train_dataset (SequenceDataset): training dataset.
        val_dataset (SequenceDataset): validation dataset.
    
    """
    # Get sorting indices of the dataset
    dataset_size = len(dataset)
    indices = jnp.arange(dataset_size)
    train_size = int(train_size * dataset_size)

    if shuffle:
        shuffled_indices = jax.random.permutation(key, indices)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
    else:
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

    # Use the PyTorch Subset class to create the training and validation datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # If only one sample in dataset, assuming the posterior testing, use the full dataset
    if len(dataset) == 1:
        train_dataset = dataset
        val_dataset = dataset
        
    return train_dataset, val_dataset


def get_sample(
        key: jax.random.PRNGKey, 
        dataset: SequenceDataset, 
        sequential: bool=False, 
        in_images: int=5, 
        resolution: int=128, 
        y_diff: bool=False
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Return a random sample from the dataset, either a sample sequence or a X, y pair of input and output.

    Args:
        key (jax.random.PRNGKey): JAX RNG key.
        dataset (SequenceDataset): SequenceDataset object.
        sequential (bool): Whether to return a sequence or a snapshot.
        in_images (int): Number of input images for the model.
        resolution (int): Resolution of the sequence data.
        y_diff (bool): Whether to predict the difference between the snapshots or the snapshots themselves.

    Returns:
        If sequential is True:
            sequence (jnp.ndarray) of shape (x, y, time, features). 
        If sequential is False:
            X (jnp.ndarray): Input sequence of shape (x, y, channel).
            y (jnp.ndarray): Ground truth sequence of shape (x, y, features).

    NOTE: features = (phi, density), channel = (x-grid, y-grid, phi(t:t+in_images), density(t:t+in_images))

    """
    # Select a random sequence from the dataset
    idx_key, time_key = jax.random.split(key)
    idx = jax.random.randint(idx_key, shape=(), minval=0, maxval=len(dataset))
    sequence = dataset[idx]
    if sequential: # return the full sequence
        return sequence
    else: # get X, y pair
        time_idx = jax.random.randint(time_key, shape=(), 
                                minval=0,
                                maxval=sequence.shape[2] - in_images - 1, 
                                )
        sequence_X = sequence[:, :, time_idx:time_idx+in_images, :]
        y = jax.lax.cond(
            jnp.any(y_diff), 
            lambda p: p - sequence_X[:, :, -1, :],
            lambda p: p,
            sequence[:, :, time_idx+in_images, :] 
        )
        X = seq_to_X(sequence_X, resolution)

        return X, y


class JAXDataLoader:
    """
    JAX-based DataLoader with optional reproducible shuffling
    
    """
    def __init__(
            self, 
            key: jax.random.PRNGKey, 
            dataset: SequenceDataset, 
            batch_size: int, 
            shuffle: bool=False):
        """
        Args:
            key (jax.random.PRNGKey): JAX RNG key.
            dataset (SequenceDataset): SequenceDataset object.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the dataset.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key

        # Generate initial indices
        self.indices = jnp.arange(len(dataset))
        if shuffle:
            self.indices = jax.random.permutation(self.key, self.indices)

    def __iter__(self)->Iterator:
        """Iterate over batches of the dataset."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield jnp.stack([self.dataset[idx] for idx in batch_indices])

    def __len__(self):
        """Number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    

@partial(jit, static_argnames=['resolution'])
def seq_to_X(input: jnp.ndarray, 
             resolution: int=128):
    """
    Transform sequence to input format of the model. Normalized fields are assumed!

    Args:
        input (jnp.ndarray): normalized input sequence of shape (x, y, t:t+in_images, feature).
        resolution (int): resolution of the sequence data.

    Returns:
        X (jnp.ndarray): transformed input sequence of shape (x, y, channel).
            -- channel shape (x-grid, y-grid, phi(t:t+in_images), density(t:t+in_images))

    """
    # get standardized mesh, assuming k0 = 0.15
    L = 2 * jnp.pi / 0.15
    grid = jnp.linspace(0, L, resolution)
    x_mesh, y_mesh = jnp.meshgrid(grid, grid, indexing='ij')
    x_mesh = x_mesh[:, :, jnp.newaxis]
    y_mesh = y_mesh[:, :, jnp.newaxis]
    x_mesh = (x_mesh - x_mesh.mean()) / x_mesh.std()
    y_mesh = (y_mesh - y_mesh.mean()) / y_mesh.std()

    # get phi and density
    phi = input[:, :, :, 0]
    density = input[:, :, :, 1]

    # concatenate the mesh and the fields
    X = jnp.concatenate([x_mesh, y_mesh, phi, density], axis=-1)

    return X