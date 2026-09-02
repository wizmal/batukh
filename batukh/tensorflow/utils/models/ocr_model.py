"""Custom convolutional-recurrent TensorFlow model for CTC OCR."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="batukh")
class ConvBlock(layers.Layer):
    """Five convolution layers followed by 2x spatial downsampling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(8, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(16, 3, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.conv4 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv5 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        return self.pool1(x)


@keras.utils.register_keras_serializable(package="batukh")
class BLSTM(layers.Layer):
    """Two stacked 256-unit bidirectional LSTM layers."""

    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.layer1 = layers.Bidirectional(
            layers.LSTM(self.units, return_sequences=True)
        )
        self.layer2 = layers.Bidirectional(
            layers.LSTM(self.units, return_sequences=True)
        )

    def call(self, inputs, training=False):
        x = self.layer1(inputs, training=training)
        return self.layer2(x, training=training)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.utils.register_keras_serializable(package="batukh")
class OCRModel(keras.Model):
    """CNN-BLSTM OCR model that returns batch-major CTC logits."""

    def __init__(self, n_classes, blank_index=None, **kwargs):
        if int(n_classes) < 2:
            raise ValueError(
                "n_classes must include at least one symbol and CTC blank."
            )
        super().__init__(**kwargs)
        self.n_classes = int(n_classes)
        self.blank_index = (
            self.n_classes - 1 if blank_index is None else int(blank_index)
        )
        if not 0 <= self.blank_index < self.n_classes:
            raise ValueError("blank_index must identify one of the output classes.")

        self.conv = ConvBlock()
        self.blstm = BLSTM()
        self.dense = layers.Dense(self.n_classes)

    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        # Treat image width as time and flatten height/channels into features.
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        feature_height, feature_channels = x.shape[2:4]
        if feature_height is None or feature_channels is None:
            raise ValueError("OCR input height and channels must be statically known.")
        feature_size = feature_height * feature_channels
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], feature_size))
        x = tf.ensure_shape(x, (None, None, feature_size))
        x = self.blstm(x, training=training)
        return self.dense(x, training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "n_classes": self.n_classes,
            "blank_index": self.blank_index,
        }
