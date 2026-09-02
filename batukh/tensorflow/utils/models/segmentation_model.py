"""Custom TensorFlow layers for document-image segmentation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="batukh")
class BottleneckUnit(layers.Layer):
    """A ResNet-style 1x1, 3x3, 1x1 bottleneck unit."""

    def __init__(
        self,
        out_filters,
        in_filters,
        stride=(1, 1),
        connect=True,
        is_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_filters = int(out_filters)
        self.in_filters = int(in_filters)
        self.stride = tuple(stride)
        self.connect = bool(connect)
        self.is_first = bool(is_first)

        self.conv1 = layers.Conv2D(
            filters=self.in_filters,
            kernel_size=1,
            strides=(1, 1),
            activation="relu",
        )
        self.conv2 = layers.Conv2D(
            filters=self.in_filters,
            kernel_size=3,
            strides=self.stride,
            padding="same",
            activation="relu",
        )
        self.conv3 = layers.Conv2D(
            filters=self.out_filters,
            kernel_size=1,
            strides=(1, 1),
        )
        self.conv4 = (
            layers.Conv2D(
                filters=self.out_filters,
                kernel_size=1,
                strides=(1, 1),
            )
            if self.connect and self.is_first
            else None
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        if not self.connect:
            return x

        shortcut = inputs
        if self.conv4 is not None:
            shortcut = self.conv4(shortcut, training=training)
        return x + shortcut

    def get_config(self):
        return {
            **super().get_config(),
            "out_filters": self.out_filters,
            "in_filters": self.in_filters,
            "stride": self.stride,
            "connect": self.connect,
            "is_first": self.is_first,
        }


@keras.utils.register_keras_serializable(package="batukh")
class ResnetLayer(layers.Layer):
    """A custom ResNet-50-shaped encoder that also returns skip tensors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=(2, 2),
            padding="same",
        )
        self.pool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.bottleneck1 = BottleneckUnit(256, 64, is_first=True)
        self.bottleneck2 = BottleneckUnit(256, 64)
        self.bottleneck3 = BottleneckUnit(256, 64, stride=(2, 2), connect=False)

        self.bottleneck4 = BottleneckUnit(512, 128, is_first=True)
        self.bottleneck5 = BottleneckUnit(512, 128)
        self.bottleneck6 = BottleneckUnit(512, 128)
        self.bottleneck7 = BottleneckUnit(512, 128, stride=(2, 2), connect=False)

        self.bottleneck8 = BottleneckUnit(1024, 256, is_first=True)
        self.bottleneck9 = BottleneckUnit(1024, 256)
        self.bottleneck10 = BottleneckUnit(1024, 256)
        self.bottleneck11 = BottleneckUnit(1024, 256)
        self.bottleneck12 = BottleneckUnit(1024, 256)
        self.bottleneck13 = BottleneckUnit(1024, 256, stride=(2, 2), connect=False)

        self.bottleneck14 = BottleneckUnit(2048, 512, is_first=True)
        self.bottleneck15 = BottleneckUnit(2048, 512)
        self.bottleneck16 = BottleneckUnit(2048, 512)

        self.conv2 = layers.Conv2D(512, kernel_size=1, padding="same")
        self.conv3 = layers.Conv2D(512, kernel_size=1, padding="same")

    def call(self, inputs, training=False):
        skip_tensors = [inputs]

        x = self.conv1(inputs, training=training)
        skip_tensors.append(x)
        x = self.pool(x)

        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        skip_tensors.append(x)

        x = self.bottleneck3(x, training=training)
        x = self.bottleneck4(x, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)
        skip_tensors.append(x)

        x = self.bottleneck7(x, training=training)
        x = self.bottleneck8(x, training=training)
        x = self.bottleneck9(x, training=training)
        x = self.bottleneck10(x, training=training)
        x = self.bottleneck11(x, training=training)
        x = self.bottleneck12(x, training=training)

        skip_tensors.append(self.conv2(x, training=training))

        x = self.bottleneck13(x, training=training)
        x = self.bottleneck14(x, training=training)
        x = self.bottleneck15(x, training=training)
        x = self.bottleneck16(x, training=training)
        x = self.conv3(x, training=training)
        return x, skip_tensors


@keras.utils.register_keras_serializable(package="batukh")
class UpScalerUnit(layers.Layer):
    """Upsample a tensor, concatenate its encoder skip, and convolve."""

    def __init__(self, in_filters, out_filters, **kwargs):
        super().__init__(**kwargs)
        self.in_filters = int(in_filters)
        self.out_filters = int(out_filters)
        self.convT = layers.Conv2DTranspose(
            self.in_filters,
            kernel_size=1,
            strides=(2, 2),
            padding="same",
        )
        self.conv = layers.Conv2D(
            self.out_filters,
            kernel_size=3,
            padding="same",
            activation="relu",
        )

    def call(self, inputs, skip, training=False):
        x = self.convT(inputs, training=training)
        # This also makes direct prediction robust to dimensions not divisible by 32.
        x = tf.image.resize(x, tf.shape(skip)[1:3], method="bilinear")
        x = tf.concat((skip, x), axis=-1)
        return self.conv(x, training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "in_filters": self.in_filters,
            "out_filters": self.out_filters,
        }


@keras.utils.register_keras_serializable(package="batukh")
class UpScaler(layers.Layer):
    """Five-stage custom decoder with encoder skip connections."""

    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = int(n_classes)
        self.upScaler1 = UpScalerUnit(512, 512)
        self.upScaler2 = UpScalerUnit(512, 256)
        self.upScaler3 = UpScalerUnit(256, 128)
        self.upScaler4 = UpScalerUnit(128, 64)
        self.upScaler5 = UpScalerUnit(64, 32)
        self.conv = layers.Conv2D(
            filters=self.n_classes,
            kernel_size=1,
            padding="same",
        )

    def call(self, inputs, skip_tensors, training=False):
        if len(skip_tensors) != 5:
            raise ValueError("UpScaler expects exactly five encoder skip tensors.")

        x = self.upScaler1(inputs, skip_tensors[4], training=training)
        x = self.upScaler2(x, skip_tensors[3], training=training)
        x = self.upScaler3(x, skip_tensors[2], training=training)
        x = self.upScaler4(x, skip_tensors[1], training=training)
        x = self.upScaler5(x, skip_tensors[0], training=training)
        return self.conv(x, training=training)

    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes}


@keras.utils.register_keras_serializable(package="batukh")
class SegmentationModel(keras.Model):
    """Custom encoder-decoder segmentation model returning per-pixel logits."""

    def __init__(self, n_classes, **kwargs):
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2.")
        super().__init__(**kwargs)
        self.n_classes = int(n_classes)
        self.resnet = ResnetLayer()
        self.upScaler = UpScaler(self.n_classes)

    def call(self, inputs, training=False):
        x, skip_tensors = self.resnet(inputs, training=training)
        return self.upScaler(x, skip_tensors, training=training)

    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes}
