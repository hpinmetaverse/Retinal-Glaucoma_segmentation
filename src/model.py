"""U-Net architecture for optic disc and cup segmentation."""

from tensorflow.keras import layers, models

from src.utils import Config


def conv_block(inputs, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def encoder_block(inputs, filters, dropout_rate=0.0):
    features = conv_block(inputs, filters, dropout_rate)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(features)
    return features, pooled


def decoder_block(inputs, skip_features, filters, dropout_rate=0.0):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, dropout_rate)
    return x


def build_unet(config: Config, base_filters=32):
    input_shape = (config.img_height, config.img_width, config.channels)
    output_channels = config.output_channels

    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, base_filters, dropout_rate=0.05)
    s2, p2 = encoder_block(p1, base_filters * 2, dropout_rate=0.05)
    s3, p3 = encoder_block(p2, base_filters * 4, dropout_rate=0.10)
    s4, p4 = encoder_block(p3, base_filters * 8, dropout_rate=0.10)

    b1 = conv_block(p4, base_filters * 16, dropout_rate=0.20)

    d1 = decoder_block(b1, s4, base_filters * 8, dropout_rate=0.10)
    d2 = decoder_block(d1, s3, base_filters * 4, dropout_rate=0.10)
    d3 = decoder_block(d2, s2, base_filters * 2, dropout_rate=0.05)
    d4 = decoder_block(d3, s1, base_filters, dropout_rate=0.05)

    outputs = layers.Conv2D(
        output_channels, kernel_size=1, activation="sigmoid", name="disc_cup_masks"
    )(d4)
    return models.Model(inputs, outputs, name="UNet_DRISTI_Disc_Cup")
