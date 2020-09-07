import tensorflow as tf
from tensorflow.keras import layers, Model


class UpSampleUnit(Model):
    def __init__(self, in_filters, out_filters):
        super(UpSampleUnit, self).__init__()

        self.convT = layers.Conv2DTranspose(
            in_filters, kernel_size=1, strides=(2, 2))
        self.conv = layers.Conv2D(
            out_filters,
            kernel_size=3,
            padding='same',
            activation='relu')

    def call(self, input_tensor, concat_tensor):
        x = self.convT(input_tensor)
        x = tf.concat([concat_tensor, x], 3)
        x = self.conv(x)
        return x


class BottleneckUnit(Model):
    def __init__(self, out_filters, in_filters, stride=(1, 1),connect=True, is_first=False):
        super(BottleneckUnit, self).__init__()
        
        self.connect=connect
        self.is_first=is_first
        

        self.conv1 = layers.Conv2D(
            filters=in_filters, kernel_size=1, strides=(
                1, 1), activation='relu')
        self.conv2 = layers.Conv2D(
            filters=in_filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            activation='relu')
        self.conv3 = layers.Conv2D(
            filters=out_filters,
            kernel_size=1,
            strides=(
                1,
                1))
        self.conv4 = layers.Conv2D(
            filters=out_filters,
            kernel_size=1,
            strides=(
                1,
                1))

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.connect:
            if self.is_first:
                input_tensor = self.conv4(input_tensor)
            return x + input_tensor
        return x

class UpSample(Model):
    def __init__(self, n_classes):
        super(UpSample,self).__init__()

        self.upsample1 = UpSampleUnit(512, 512)
        self.upsample2 = UpSampleUnit(512, 256)
        self.upsample3 = UpSampleUnit(256, 128)
        self.upsample4 = UpSampleUnit(128, 64)
        self.upsample5 = UpSampleUnit(64, 32)
        self.conv = layers.Conv2D(
            filters=n_classes,
            kernel_size=1,
            padding='same')

    def call(self, input_tensor, concat_tensors):
        x = self.upsample1.call(input_tensor, concat_tensors.pop(-1))
        x = self.upsample2.call(x, concat_tensors.pop(-1))
        x = self.upsample3.call(x, concat_tensors.pop(-1))
        x = self.upsample4.call(x, concat_tensors.pop(-1))
        x = self.upsample5.call(x, concat_tensors.pop(-1))
        x = self.conv(x)
        return x



class ResnetLayer(Model):
    def __init__(self):
        super(ResnetLayer, self).__init__()

        self.conv1 = layers.Conv2D(
            filters=64, kernel_size=7, strides=(
                2, 2), padding="same")
        self.pool = layers.MaxPool2D(strides=(2, 2))

        self.bottleneck1 = BottleneckUnit(256, 64, is_first=True)
        self.bottleneck2 = BottleneckUnit(256, 64)
        self.bottleneck3 = BottleneckUnit(256, 64, (2, 2), connect = False)

        self.bottleneck4 = BottleneckUnit(512, 128, is_first=True)
        self.bottleneck5 = BottleneckUnit(512, 128)
        self.bottleneck6 = BottleneckUnit(512, 128)
        self.bottleneck7 = BottleneckUnit(512, 128, (2, 2), connect=False)

        self.bottleneck8 = BottleneckUnit(1024, 256, is_first=True)
        self.bottleneck9 = BottleneckUnit(1024, 256)
        self.bottleneck10 = BottleneckUnit(1024, 256)
        self.bottleneck11 = BottleneckUnit(1024, 256)
        self.bottleneck12 = BottleneckUnit(1024, 256)
        self.bottleneck13 = BottleneckUnit(1024, 256, (2, 2), connect=False)

        self.bottleneck14 = BottleneckUnit(2048, 512, is_first=True)
        self.bottleneck15 = BottleneckUnit(2048, 512)
        self.bottleneck16 = BottleneckUnit(2048, 512)

        self.conv2 = layers.Conv2D(
            512, kernel_size=1, strides=(
                1, 1), padding="same")
        self.conv3 = layers.Conv2D(
            512, kernel_size=1, strides=(
                1, 1), padding="same")

    def call(self, input_tensor):
        concat_tensors = []
        concat_tensors.append(input_tensor)
        x = self.conv1(input_tensor)
        concat_tensors.append(x)

        x = self.pool(x)

        x = self.bottleneck1.call(x)
        x = self.bottleneck2.call(x)
        concat_tensors.append(x)
        x = self.bottleneck3.call(x)
        x = self.bottleneck4.call(x)
        x = self.bottleneck5.call(x)
        x = self.bottleneck6.call(x)
        concat_tensors.append(x)
        x = self.bottleneck7.call(x)
        x = self.bottleneck8.call(x)
        x = self.bottleneck9.call(x)
        x = self.bottleneck10.call(x)
        x = self.bottleneck11.call(x)
        x = self.bottleneck12.call(x)

        y = self.conv2(x)
        concat_tensors.append(y)

        x = self.bottleneck13.call(x)
        x = self.bottleneck14.call(x)
        x = self.bottleneck15.call(x)
        x = self.bottleneck16.call(x)

        x = self.conv3(x)
        return x, concat_tensors


class SegmentationModel(Model):
    def __init__(self, n_classes=2):
        super(SegmentationModel,self).__init__()
        self.resnet = ResnetLayer()
        self.upsample = UpSample(n_classes)

    def call(self, input_tensor):
        x, concat_tensors = self.resnet.call(input_tensor)
        x = self.upsample.call(x, concat_tensors)
        return x

