import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBlock(Model):

    def __init__(self):
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
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool1(x)
        return x


class BLSTM(Model):
    def __init__(self):
        self.layer1 = layers.Bidirectional(
            layers.LSTM(units=256, return_sequences=True))

    def call(self, input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer1(x)
        return x


class OCRModel(Model):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.conv = ConvBlock()
        self.blstm = BLSTM()
        self.dense = layers.Dense(units=self.n_classes)

    def call(self, input_tensor):
        x = self.conv.call(input_tensor)
        x = tf.transpose(x, (0, 2, 1, 3))
        x = layers.Reshape((-1, 16*128))(x)
        x = self.blstm.call(x)
        x = self.dense(x)
        return x
