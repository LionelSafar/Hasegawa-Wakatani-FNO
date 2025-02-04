import flax.linen as nn

# This file contains a 2D ResNet implementation.

class ResNetBlock(nn.Module):
    """
    ResNet block with optional downsampling for 2D images with periodic boundary conditions.
    
    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)
    
    Args:
        out_channels: number of output channels
        downsample: whether to downsample the input tensor

    """
    out_channels: int
    downsample: bool = False

    @nn.compact
    def __call__(self, x, train):
        residue = x
        x = nn.Conv(self.out_channels, 
                    kernel_size=(3, 3), 
                    padding='CIRCULAR',
                    strides=(1, 1) if not self.downsample else (2, 2),
                    kernel_init=nn.initializers.variance_scaling(2.0, 
                                mode='fan_out', distribution='normal'), 
                    use_bias=False)(x)
        x = nn.BatchNorm(axis_name='batch')(x, use_running_average=not train)
        x = nn.gelu(x)
        x = nn.Conv(self.out_channels, 
                    kernel_size=(3, 3), 
                    padding='CIRCULAR',
                    strides=(1, 1),
                    kernel_init=nn.initializers.variance_scaling(2.0, 
                                mode='fan_out', distribution='normal'), 
                    use_bias=False)(x)
        x = nn.BatchNorm(axis_name='batch')(x, use_running_average=not train)

        # If downsampling, adapt the residue to the downsampled shape as well
        if self.downsample:
            residue = nn.Conv(self.out_channels, 
                              kernel_size=(1, 1), 
                              strides=(2, 2), 
                              use_bias=False,
                              kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal'))(residue)
        else:
            residue = nn.Conv(self.out_channels, 
                              kernel_size=(1, 1), 
                              use_bias=False,
                              kernel_init=nn.initializers.variance_scaling(2.0, 
                                        mode='fan_out', distribution='normal'))(residue)
        x = x + residue
        x = nn.gelu(x)
        return x

class ResNet(nn.Module):
    """
    ResNet model for 2D images with periodic boundary conditions.

    input tensor format: (x_dim, y_dim, in_channels)
    output tensor format: (x_dim, y_dim, out_channels)

    Args:
        out_channels: number of output channels
        hidden_channels: tuple of hidden channels for each group
        num_blocks: tuple of number of ResNet blocks per group
        downsample: whether to downsample the input tensor of each first block in a group 
            (except first group)

    """
    out_channels: int
    hidden_channels: tuple #e.g. (16, 32, 64)
    num_blocks: tuple # e.g (8, 8, 8)
    downsample: bool = False

    @nn.compact
    def __call__(self, x, train):
        # Lifting layer
        x = nn.Conv(self.hidden_channels[0], 
                    kernel_size=(3, 3), 
                    padding='CIRCULAR',
                    kernel_init=nn.initializers.variance_scaling(2.0, 
                            mode='fan_out', distribution='normal'), 
                    use_bias=False)(x)
        x = nn.BatchNorm(axis_name='batch')(x, use_running_average=not train)
        x = nn.gelu(x)

        # ResNet blocks
        for i, blocks in enumerate(self.num_blocks):
            for block in range(blocks):
                # Downsample the input tensor of the first block in each non-first group
                downsample = (i > 0 and block == 0) and self.downsample
                x = ResNetBlock(self.hidden_channels[i], downsample)(x, train=train)

        # Projection layer
        x = nn.Conv(self.out_channels, 
                    kernel_size=(1, 1),
                    kernel_init=nn.initializers.variance_scaling(2.0, 
                            mode='fan_out', distribution='normal'), 
                    use_bias=False)(x)
        return x