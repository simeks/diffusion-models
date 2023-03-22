import tensorflow as tf

from tensorflow import keras
from keras import activations, layers

def block(
    width,
    activation=activations.relu,
    use_batch_norm=True,
    dropout_rate=0.0,
    ):
    """ 2D Convolutional block
    """
    def apply(x):
        x = layers.Conv2D(width, kernel_size=3, padding='same', activation=activation)(x)
    
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv2D(width, kernel_size=3, padding='same', activation=activation)(x)

        return x

    return apply

def down_block(
    width,
    block_depth,
    pooling_layer=layers.MaxPool2D
    ):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = block(width)(x)
            skips.append(x)
        x = pooling_layer(pool_size=2)(x)
        return x

    return apply

def up_block(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = block(width)(x)
        return x

    return apply

def unet(
    image_size: int,
    num_channels: int,
    widths: list[int] = [32, 64, 128],
    block_depth: int = 2,
):
    """ U-Net [1] model

    Args:
        image_size (int): Image size
        num_channels: Number of channels
        widths: Widths of the U-Net
        block_depth: Depth of the U-Net blocks
    
    References
        [1] https://arxiv.org/abs/1505.04597
    """
    input = layers.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(num_channels, kernel_size=1)(input)
    
    skips = []

    # Encoder
    for width in widths[:-1]:
        x = down_block(width, block_depth)([x, skips])

    # Bottleneck
    for _ in range(block_depth):
        x = block(widths[-1])(x)

    # Decoder
    for width in reversed(widths[:-1]):
        x = up_block(width, block_depth)([x, skips])

    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer='zeros')(x)

    return tf.keras.Model(input, x, name='unet')
