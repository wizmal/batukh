import numpy as np
import tensorflow as tf
from tensorflow import keras

from batukh.tensorflow.ocr import OCR
from batukh.tensorflow.segmenter import PageExtractor
from batukh.tensorflow.utils.train import Train


class TinySegmentationModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.n_classes = 2
        self.output_layer = keras.layers.Conv2D(2, kernel_size=1)

    def call(self, inputs, training=False):
        return self.output_layer(inputs, training=training)


class TinyLoader:
    def __init__(self):
        self.images = tf.random.uniform((2, 8, 8, 3), seed=7)
        ids = tf.cast(self.images[..., 0] > 0.5, tf.int32)
        self.labels = tf.one_hot(ids, 2)

    def __len__(self):
        return len(self.images)

    def __call__(self, batch_size=1, repeat=1, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        if shuffle:
            dataset = dataset.shuffle(len(self), seed=7)
        return dataset.repeat(repeat).batch(batch_size)


def test_custom_trainer_updates_weights_and_saves_keras3_weights(tmp_path):
    model = TinySegmentationModel()
    sample = tf.zeros((1, 8, 8, 3))
    model(sample)
    before = [weight.numpy().copy() for weight in model.trainable_variables]
    trainer = Train(model, log_dir=tmp_path / "logs")

    history = trainer.train(
        1,
        train_dl=TinyLoader(),
        batch_size=2,
        save_checkpoints=False,
        log_freq=None,
    )
    saved_path = trainer.save_model(tmp_path / "tiny")

    assert np.isfinite(history["loss"][0])
    assert any(
        not np.array_equal(previous, current.numpy())
        for previous, current in zip(before, model.trainable_variables)
    )
    assert str(saved_path).endswith(".weights.h5")
    assert saved_path.exists()

    restored_model = TinySegmentationModel()
    restored_model(sample)
    restored_trainer = Train(restored_model, log_dir=tmp_path / "restored-logs")
    restored_trainer.load_model(saved_path)
    for expected, restored in zip(model.weights, restored_model.weights):
        np.testing.assert_allclose(expected.numpy(), restored.numpy())

    legacy_path = tmp_path / "legacy-checkpoint"
    tf.train.Checkpoint(model=model).write(str(legacy_path))
    legacy_model = TinySegmentationModel()
    legacy_model(sample)
    legacy_trainer = Train(legacy_model, log_dir=tmp_path / "legacy-logs")
    legacy_trainer.load_model(legacy_path)
    for expected, restored in zip(model.weights, legacy_model.weights):
        np.testing.assert_allclose(expected.numpy(), restored.numpy())


def test_ocr_greedy_decode_uses_final_class_as_blank(tmp_path):
    ocr = OCR(n_classes=3, log_dir=tmp_path / "logs")
    class_ids = tf.constant([[0, 0, 2, 1]], dtype=tf.int32)
    logits = tf.one_hot(class_ids, depth=3, on_value=10.0, off_value=-10.0)

    decoded = ocr.decode(logits, table=["a", "b", "<blank>"])

    assert decoded == ["ab"]


def test_ocr_wrapper_runs_a_real_ctc_training_step(tmp_path):
    images_dir = tmp_path / "ocr" / "images"
    images_dir.mkdir(parents=True)
    white = tf.constant(255, dtype=tf.uint8)
    tf.io.write_file(
        str(images_dir / "0.png"),
        tf.io.encode_png(tf.fill((20, 40, 1), white)),
    )
    tf.io.write_file(
        str(images_dir / "1.png"),
        tf.io.encode_png(tf.fill((20, 60, 1), white)),
    )
    (tmp_path / "ocr" / "labels.txt").write_text(
        "0: ab\n1: ba\n",
        encoding="utf-8",
    )
    (tmp_path / "ocr" / "table.txt").write_text(
        "a\nb\n<blank>\n",
        encoding="utf-8",
    )
    ocr = OCR(n_classes=3, log_dir=tmp_path / "logs")
    ocr.load_data(tmp_path / "ocr", height=16)

    history = ocr.train(
        1,
        batch_size=2,
        save_checkpoints=False,
        log_freq=None,
    )

    assert np.isfinite(history["loss"][0])


def test_page_postprocessing_scales_boxes_to_original_image(tmp_path):
    extractor = PageExtractor(log_dir=tmp_path / "logs")
    class_map = np.zeros((4, 4), dtype=np.int32)
    class_map[1:3, 1:3] = 1
    logits = tf.one_hot(class_map, depth=2, on_value=10.0, off_value=-10.0)
    original = np.zeros((8, 8, 3), dtype=np.uint8)

    pages = extractor.get_pages(original, logits, iterations=0)

    assert len(pages) == 1
    assert pages[0].shape == (4, 4, 3)
