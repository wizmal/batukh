import tensorflow as tf
from tensorflow import keras

from batukh.tensorflow.utils.models.ocr_model import OCRModel
from batukh.tensorflow.utils.models.segmentation_model import SegmentationModel


def test_segmentation_model_preserves_image_size_without_sequential_layers():
    model = SegmentationModel(n_classes=3)

    output = model(tf.random.uniform((1, 32, 48, 3)))

    assert output.shape == (1, 32, 48, 3)
    assert not any(isinstance(layer, keras.Sequential) for layer in model.layers)


def test_ocr_model_uses_width_as_ctc_time_axis():
    model = OCRModel(n_classes=5)

    output = model(tf.random.uniform((2, 16, 25, 1)))

    assert output.shape == (2, 13, 5)
    assert model.blank_index == 4


def test_ocr_model_traces_with_variable_width():
    model = OCRModel(n_classes=5)

    @tf.function(input_signature=(tf.TensorSpec((None, 16, None, 1), tf.float32),))
    def run(inputs):
        return model(inputs, training=False)

    first = run(tf.random.uniform((1, 16, 25, 1)))
    second = run(tf.random.uniform((2, 16, 41, 1)))

    assert first.shape == (1, 13, 5)
    assert second.shape == (2, 21, 5)
