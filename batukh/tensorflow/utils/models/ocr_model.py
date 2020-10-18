import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBlock(Model):
    r""" This block consists of five :class:`tensorflow.keras.layers.Conv2D` convolution layers and a :class:`tensorflow.keras.layers.MaxPool2D` maxpool layer.
        Convolution layers is of  kernel size  :math:`3` ,same padding and relu activation, and filters :math:`8 , 16 , 32 , 64` and :math:`128` respectively.
        Maxpool layer is of pool size :math:`(2 , 2)`, strides :math:`( 2 , 2)` ans same padding.


    """

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=8, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(
            filters=16, kernel_size=3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(
            filters=32, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(
            filters=64, kernel_size=3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(
            filters=128, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(
            2, 2), strides=(2, 2), padding='same')

    def call(self, input_tensor):
        r"""

        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.

        Returns:
            :class:`tensorflow.Tensor` : Output image tensor.
        """
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool1(x)
        return x


class BLSTM(Model):
    r"""Consists of a two :class:`tensorflow.keras.layers.Bidirectional` layer with :math:`256` units.

    """

    def __init__(self):
        super(BLSTM, self).__init__()

        self.layer1 = layers.Bidirectional(
            layers.LSTM(units=256, return_sequences=True))
        self.layer2 = layers.Bidirectional(
            layers.LSTM(units=256, return_sequences=True))

    def call(self, input_tensor):
        r"""
        Args:
            input_tensor( :class:`tensorflow.Tensor`): Input tensor.

        Returns:
            :class:`tensorflow.Tensor` : Output tensor.
        """
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        return x


class OCRModel(Model):
    r"""Consistis of :class:`~batukh.tensorflow.utils.models.ocr_model.ConvBlock` block, :class:`~batukh.tensorflow.utils.models.ocr_model.BLSTM` block and :class:`tensorflow.keras.layers.dense` layer.

    Args:
        n_classes (int)  : Specifies number of output classes in :class:`tensorflow.keras.layers.dense` layer.
    """

    def __init__(self, n_classes):
        super(OCRModel, self).__init__()
        self.n_classes = n_classes
        self.conv = ConvBlock()
        self.blstm = BLSTM()
        self.dense = layers.Dense(units=self.n_classes)

    def call(self, input_tensor):
        r"""
        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.

        Returns:
            :class:`tensorflow.Tensor` : Output Tensor."""
        x = self.conv.call(input_tensor)
        x = tf.transpose(x, (0, 2, 1, 3))
        x = layers.Reshape((-1, x.shape[-2]*128))(x)
        x = self.blstm.call(x)
        x = self.dense(x)
        return x
