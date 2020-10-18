import tensorflow as tf
from tensorflow.keras import layers, Model


class BottleneckUnit(Model):
    r"""Consists of :math:`3` :class:`tensorflow.keras.layers.Conv2D` convolution layers with same padding,
    relu activation and kernel size :math:`1,3,1` respectively.

    Args:
        out_filters (int)         : Specifies filters of third convulution layer.
        in_filters (int)          : Specifies fliters of first and second convulution layer.
        strides (tuple,optional)  : Specifies strides of convulution layers.
            Default : :math:`(1,1)`
        connect (bol,optional) : Specifies if the layer will be used for horizantal connection in upscaler.
            Default :``True``.
        is_first (bol,optional): Specifies if the bottleneck unit is first of the bollteneck block.
            Default :``False``.


    """

    def __init__(self, out_filters, in_filters, stride=(1, 1), connect=True, is_first=False):
        super(BottleneckUnit, self).__init__()

        self.connect = connect
        self.is_first = is_first

        self.conv1 = layers.Conv2D(
            filters=in_filters, kernel_size=1, strides=(
                1, 1), activation='relu')
        self.conv2 = layers.Conv2D(
            filters=in_filters, kernel_size=3, strides=stride, padding="same", activation='relu')
        self.conv3 = layers.Conv2D(
            filters=out_filters, kernel_size=1, strides=(1, 1))
        self.conv4 = layers.Conv2D(
            filters=out_filters, kernel_size=1, strides=(1, 1))

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
        if self.connect:
            if self.is_first:
                input_tensor = self.conv4(input_tensor)
            return x + input_tensor
        return x


class ResnetLayer(Model):
    r"""Consists of three :class:`tensorflow.keras.layers.conv2D` convolution layer, 
    :class:`tensorflow.keras.layers.MaxPool2D` maxpoll layer and :math:`16`  
    :class:`~batukh.tensorflow.utils.models.segmentation_model.BottleneckUnit` units.

    """

    def __init__(self):
        super(ResnetLayer, self).__init__()

        self.conv1 = layers.Conv2D(
            filters=64, kernel_size=7, strides=(
                2, 2), padding="same")
        self.pool = layers.MaxPool2D(strides=(2, 2))

        self.bottleneck1 = BottleneckUnit(256, 64, is_first=True)
        self.bottleneck2 = BottleneckUnit(256, 64)
        self.bottleneck3 = BottleneckUnit(
            256, 64, stride=(2, 2), connect=False)

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
        r"""
        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.

        Returns:
            :class:`tensorflow.Tensor`        : Output image tensor.
            list : List of image tensor used in horizantal connections of upscaller.
        """
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


class UpScalerUnit(Model):
    r"""Consists of :class:`layers.Conv2DT` convolution transpose layer and
    :class:`tensorflow.keras.layers.Conv2D` convolution layer.

    Args:
        in_filters (int)  : Specifies the filters of convolution transpose layer.
        out_filters (int) : Specifies the filters of convolution layer.

    """

    def __init__(self, in_filters, out_filters):
        super(UpScalerUnit, self).__init__()

        self.convT = layers.Conv2DTranspose(
            in_filters, kernel_size=1, strides=(2, 2))
        self.conv = layers.Conv2D(
            out_filters,
            kernel_size=3,
            padding='same',
            activation='relu')

    def call(self, input_tensor, concat_tensor):
        r"""
        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.
            concat_tensor ( :class:`tensorflow.Tensor`): Image tensor used to concat with the input tensor.

        Returns:
            :class:`tensorflow.Tensor` : Output image tensor.
        """
        x = self.convT(input_tensor)
        x = tf.concat([concat_tensor, x], 3)
        x = self.conv(x)
        return x


class UpScaler(Model):
    r"""Consists of :class:`tensorflow.keras.layers.Conv2D` convolution layer and 
    five :class:`~batukh.tensorflow.utils.models.segmentation_model.UpScalerUnit`.

    Args:
        n_classes (int) : Specifies the number of classes or channels in output image.
    """

    def __init__(self, n_classes):
        super(UpScaler, self).__init__()

        self.upScaler1 = UpScalerUnit(512, 512)
        self.upScaler2 = UpScalerUnit(512, 256)
        self.upScaler3 = UpScalerUnit(256, 128)
        self.upScaler4 = UpScalerUnit(128, 64)
        self.upScaler5 = UpScalerUnit(64, 32)
        self.conv = layers.Conv2D(
            filters=n_classes,
            kernel_size=1,
            padding='same')

    def call(self, input_tensor, concat_tensors):
        r"""
        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.
            concat_tensors (list)    : List of concat tensors used for horizantal connections.

        Returns:
            :class:`tensorflow.Tensor` : Output image tensor with channels equal to n_classes.
            """
        x = self.upScaler1.call(input_tensor, concat_tensors.pop(-1))
        x = self.upScaler2.call(x, concat_tensors.pop(-1))
        x = self.upScaler3.call(x, concat_tensors.pop(-1))
        x = self.upScaler4.call(x, concat_tensors.pop(-1))
        x = self.upScaler5.call(x, concat_tensors.pop(-1))
        x = self.conv(x)
        return x


class SegmentationModel(Model):
    r"""Consists of :class:`~batukh.tensorflow.utils.models.segmentation_model.ResnetLayer` and  
    :class:`~batukh.tensorflow.utils.models.segmentation_model.UpScaler`.

    Args:
        n_classes (int) : Specifies number of classes or channels in output image tensor.

    """

    def __init__(self, n_classes):
        super(SegmentationModel, self).__init__()
        self.n_classes = n_classes
        self.resnet = ResnetLayer()
        self.upScaler = UpScaler(self.n_classes)

    def call(self, input_tensor):
        r"""
        Args:
            input_tensor ( :class:`tensorflow.Tensor`) : Input image tensor.

        Returns:
            :class:`tensorflow.Tensor` : Output image tensor with channels equal to n_classes.
        """
        x, concat_tensors = self.resnet.call(input_tensor)
        x = self.upScaler.call(x, concat_tensors)
        return x
