import jax.numpy as jnp
import flax.linen as nn

# This file contains a 2D U-Net implementation.

class ConvBlock2D(nn.Module):
    """
    Convolutional U-Net block for 2D images with periodic boundary conditions.
    Consists of two convolutional layers with batch normalization and GELU activation.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        out_channels: number of output channels
    
    """
    out_channels: int

    @nn.compact
    def __call__(self, x, train):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), 
                    padding='SAME', kernel_init=nn.initializers.variance_scaling(2.0, 
                        mode='fan_out', distribution='normal'),
                    use_bias=False)(x)
        x = nn.BatchNorm(axis_name='batch')(x, use_running_average=not train)
        x = nn.gelu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), 
                    padding='SAME', kernel_init=nn.initializers.variance_scaling(2.0, 
                        mode='fan_out', distribution='normal'),
                    use_bias=False)(x)
        x = nn.BatchNorm(axis_name='batch')(x, use_running_average=not train)
        x = nn.gelu(x)
        return x



class Unet(nn.Module):
    """
    U-Net model for 2D images, instead of the more common
    resolution reduction due to 3x3 convs, we use periodic padding.

    Encoder blocks consist of a ConvBlock2D followed by max pooling.
    Decoder blocks consist of a ConvBlock2D followed by upsampling with 2x2 transposed convolution.
    Each up-/downsampling step halves the hidden dimensionality.
    Bottleneck consists of a simple ConvBlock2D.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        initial_hidden_channels: number of hidden channels in the first layer
        out_channels: number of output channels
        depth: number of encoder/decoder blocks

    """
    initial_hidden_channels: int
    out_channels: int
    depth: int

    @nn.compact
    def __call__(self, x, train):
        skip_connections = [] # keep track of skip connections for concatenation

        # Contracting path
        for i in range(self.depth):
            hidden_channels = self.initial_hidden_channels * (2**i)
            x = ConvBlock2D(hidden_channels)(x, train)
            skip_connections.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        # Bottleneck
        x = ConvBlock2D(self.initial_hidden_channels*(2**self.depth))(x, train)

        # Expanding path
        for i in range(self.depth):
            j = i + 1
            hidden_channels = self.initial_hidden_channels * (2**(self.depth - j))
            x = nn.ConvTranspose(hidden_channels, 
                                 kernel_size=(3, 3), 
                                 strides=(2, 2), 
                                 padding='CIRCULAR',
                                 kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal'),
                                 use_bias=False)(x)
            x = jnp.concatenate([x, skip_connections.pop()], axis=-1) # add skip connection
            x = ConvBlock2D(hidden_channels)(x, train)

        # Projection layer
        x = nn.Conv(self.out_channels, 
                    kernel_size=(1, 1), 
                    kernel_init=nn.initializers.variance_scaling(2.0, 
                            mode='fan_out', distribution='normal'),
                    use_bias=False)(x)
        return x



