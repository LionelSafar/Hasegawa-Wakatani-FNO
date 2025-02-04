import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable, List

from modules.Unet import Unet


# This file contains the Flax modules for the FNO and U-FNO Neural Networks.

class SpectralConv2D(nn.Module):
    """
    Spectral Convolution Layer for 2D images.
    The layer computes a 2D Fourier transform, multiplies the weights in Fourier space and
    then computes the inverse Fourier transform.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        modes: tuple with maximum frequency components in x and y direction

    """
    in_channels: int
    out_channels: int
    modes: Tuple[int, int]

    def setup(self):
        # Initialize real and imaginary weights based on modified Xavier initialization
        scale = jnp.sqrt(2 / (self.in_channels + self.out_channels))
        self.real_weights = self.param('real_weights', 
                                       lambda key, shape: jax.random.normal(key, shape) * scale,
                                       (self.in_channels, self.out_channels, 2*self.modes[0], self.modes[1])
                                       )
        self.imag_weights = self.param('imag_weights',
                                        lambda key, shape: jax.random.normal(key, shape) * scale,
                                        (self.in_channels, self.out_channels, 2*self.modes[0], self.modes[1])
                                        )

    def __call__(self, x):
        # Assemble the complex weight tensor R
        R = self.real_weights + 1j * self.imag_weights

        x_dim, y_dim, _ = x.shape
        if self.modes[1]>x_dim//2+1: # Raise error if modes is chosen too large for the resolution
            raise ValueError("truncation mode is greater than the number of Fourier modes in y direction")

        #1) 2D Fourier transform and mode truncation
        x_hat = jnp.fft.rfft2(x, axes=(0, 1)) # along y and x dimensions
        pos_modes = x_hat[:self.modes[0], :self.modes[1], :]
        neg_modes = x_hat[-self.modes[0]:, :self.modes[1], :]
        x_hat_under_modes = jnp.concatenate([pos_modes, neg_modes], axis=0)

        #2) Multiply with weights and set high frequencies to zero, use einsum
        #NOTE: i = input channel, o = output channel, y = y-mode, x = x-mode
        out_hat_under_modes = jnp.einsum('xyi, ioxy->xyo', x_hat_under_modes, R)
        out_hat = jnp.zeros((x_hat.shape[0], x_hat.shape[1], self.out_channels), 
                             dtype=x_hat.dtype)
        pos_modes = out_hat_under_modes[:self.modes[0], :self.modes[1], :]
        neg_modes = out_hat_under_modes[-self.modes[0]:, :self.modes[1], :]
        out_hat = out_hat.at[:self.modes[0], :self.modes[1], :].set(pos_modes)
        out_hat = out_hat.at[-self.modes[0]:, :self.modes[1], :].set(neg_modes)

        #3) Inverse Fourier transform
        out = jnp.fft.irfft2(out_hat, s=[x_dim, y_dim], axes=(0, 1))
        return out



class FNOLayer2D(nn.Module):
    """
    Fourier Layer class for 2DFNO. 
    Computes the spectral convolution and bypass convolution, applying batch normalization and activation.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        modes: tuple with maximum frequency components in x and y direction
        activation: activation function

    """
    in_channels: int
    out_channels: int
    modes: Tuple[int, int]
    activation: Callable 

    def setup(self):
        #Initialize conv layer based on variance scaling
        self.spectral_conv = SpectralConv2D(self.in_channels, 
                                            self.out_channels, 
                                            self.modes)
        self.bypass_conv = nn.Conv(self.out_channels,
                                    kernel_size=(1, 1),
                                    kernel_init=nn.initializers.variance_scaling(2.0, 
                                            mode='fan_out', distribution='normal'), 
                                    use_bias=False)
        self.batchnorm = nn.BatchNorm(axis_name='batch') 
        #NOTE: axis_name='batch' is used for Flax to recognize the batch dimension to vectorize over

    def __call__(self, x, train):
        x = self.spectral_conv(x) + self.bypass_conv(x)
        x = self.batchnorm(x, use_running_average=not train)
        x = self.activation(x)
        return x
    


class FNO2D(nn.Module):
    """
    Full FNO2D model.
    Consists of a lifting layer, n FNO layers, and a projection layer.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        modes: tuple with maximum frequency components in x and y direction
        hidden_channels: number of hidden channels
        n_layers: number of FNO layers

    """
    in_channels: int
    out_channels: int
    modes: Tuple[int]
    hidden_channels: int
    n_layers: int
    fno_layers = List[FNOLayer2D]

    def setup(self):
        # instantiate lifting layer
        self.lifting = nn.Conv(features=self.hidden_channels,
                               kernel_size=(1, 1), 
                               use_bias=False,
                               kernel_init=nn.initializers.variance_scaling(2.0, 
                                    mode='fan_out', distribution='normal')
                               )
        
        self.fno_layers = [FNOLayer2D(in_channels=self.hidden_channels, 
                            out_channels=self.hidden_channels, 
                            modes=self.modes, 
                            activation=nn.gelu) for _ in range(self.n_layers)]

        self.projection = nn.Conv(features=self.out_channels, padding='CIRCULAR',
                                  kernel_size=(1, 1), 
                                  use_bias=False,
                                  kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal')
                                  )

    def __call__(self, x, train: bool):
        x = self.lifting(x)
        for fno_layer in self.fno_layers:
            x = fno_layer(x, train)
        x = self.projection(x)
        return x
    


class UFNO_Layer2D(nn.Module):
    """
    U-FNO Layer class for 2D U-FNO.
    U-FNO applies a U-Net to the small scale flow and adds it to the output of the spectral convolution.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        modes: tuple with maximum frequency components in x and y direction

    """
    in_channels: int
    out_channels: int
    modes: Tuple[int, int]
    activation: Callable 

    @nn.compact
    def __call__(self, x, train):
        x_hat = SpectralConv2D(self.in_channels, 
                               self.out_channels, 
                               self.modes)(x)
        x_bypass = nn.Conv(self.out_channels, 
                           kernel_size=(1, 1), 
                           kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal'), 
                           use_bias=False)(x)
        s = x - x_hat # small scale flow
        x_unet = Unet(self.in_channels, self.out_channels, 2)(s, train)
        x = x_hat + x_unet + x_bypass
        x = nn.BatchNorm(axis_name='batch', momentum=0.9, epsilon=1e-5)(x, use_running_average=not train)
        x = nn.gelu(x)
        return x
        


class UFNO2D(nn.Module):
    """
    Full U-FNO2D model.
    Consists of a lifting layer, n U-FNO layers, and a projection layer.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        modes: tuple with maximum frequency components in x and y direction
        hidden_channels: number of hidden channels
        n_layers: number of U-FNO layers

    """
    in_channels: int
    out_channels: int
    modes: Tuple[int]
    hidden_channels: int
    n_layers: int
    ufno_layers = List[UFNO_Layer2D]

    def setup(self):
        self.lifting = nn.Conv(features=self.hidden_channels,
                               kernel_size=(1, 1), 
                               use_bias=False,
                               kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal')
                              )
        self.ufno_layers = [UFNO_Layer2D(in_channels=self.hidden_channels, 
                            out_channels=self.hidden_channels, 
                            modes=self.modes, 
                            activation=nn.gelu) for _ in range(self.n_layers)]
        self.projection = nn.Conv(features=self.out_channels,
                                  kernel_size=(1, 1), 
                                  use_bias=False,
                                  kernel_init=nn.initializers.variance_scaling(2.0, 
                                            mode='fan_out', distribution='normal')
                                )

    def __call__(self, x, train: bool):
        x = self.lifting(x)
        for ufno_layer in self.ufno_layers:
            x = ufno_layer(x, train)
        x = self.projection(x)
        return x
    