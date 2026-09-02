import tensorflow as tf

from batukh.tensorflow.utils.data.augmentation import MultipleColorJitter
from batukh.tensorflow.utils.data.dataloader import (
    OCRDataLoader,
    SegmentationDataLoader,
)


def _write_png(path, tensor):
    tf.io.write_file(str(path), tf.io.encode_png(tensor))


def test_segmentation_loader_pairs_resizes_and_pads_images(tmp_path):
    originals = tmp_path / "originals"
    labels = tmp_path / "labels"
    originals.mkdir()
    labels.mkdir()

    image_a = tf.zeros((35, 67, 3), dtype=tf.uint8)
    red_channel = tf.concat(
        (
            tf.fill((35, 20, 1), tf.constant(255, tf.uint8)),
            tf.zeros((35, 47, 1), dtype=tf.uint8),
        ),
        axis=1,
    )
    label_a = tf.concat(
        (red_channel, tf.zeros((35, 67, 2), tf.uint8)),
        axis=-1,
    )
    image_b = tf.zeros((70, 35, 3), dtype=tf.uint8)
    label_b = tf.zeros((70, 35, 3), dtype=tf.uint8)

    _write_png(originals / "a.png", image_a)
    _write_png(labels / "a.png", label_a)
    _write_png(originals / "b.png", image_b)
    _write_png(labels / "b.png", label_b)

    loader = SegmentationDataLoader(tmp_path, n_classes=2)
    images, masks = next(iter(loader(batch_size=2)))

    assert len(loader) == 2
    assert images.shape == (2, 64, 64, 3)
    assert masks.shape == (2, 64, 64, 2)
    assert tf.reduce_max(masks[..., 1]).numpy() == 1.0


def test_segmentation_loader_preserves_multiclass_ids(tmp_path):
    originals = tmp_path / "originals"
    labels = tmp_path / "labels"
    originals.mkdir()
    labels.mkdir()
    _write_png(originals / "page.png", tf.zeros((32, 32, 3), tf.uint8))
    class_ids = tf.fill((32, 32, 1), tf.constant(2, tf.uint8))
    mask = tf.concat((class_ids, tf.zeros((32, 32, 2), tf.uint8)), axis=-1)
    _write_png(labels / "page.png", mask)

    loader = SegmentationDataLoader(tmp_path, n_classes=3)
    _, encoded = next(iter(loader()))

    assert tf.reduce_all(tf.argmax(encoded, axis=-1) == 2).numpy()


def test_ocr_loader_builds_padded_images_and_sparse_labels(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    white = tf.constant(255, dtype=tf.uint8)
    _write_png(images_dir / "0.png", tf.fill((20, 40, 1), white))
    _write_png(images_dir / "1.png", tf.fill((20, 60, 1), white))
    (tmp_path / "labels.txt").write_text("0: ab\n1: ba\n", encoding="utf-8")
    (tmp_path / "table.txt").write_text("a\nb\n<blank>\n", encoding="utf-8")

    loader = OCRDataLoader(tmp_path, height=32)
    images, (labels, logit_lengths) = next(iter(loader(batch_size=2)))

    assert images.shape == (2, 32, 96, 1)
    assert labels.dense_shape.numpy().tolist() == [2, 2]
    assert logit_lengths.numpy().tolist() == [32, 48]
    assert tf.sparse.to_dense(
        labels, default_value=loader.blank_index
    ).numpy().tolist() == [
        [0, 1],
        [1, 0],
    ]


def test_color_jitter_keeps_labels_and_unselected_images_unchanged():
    images = tf.ones((2, 4, 4, 3), tf.float32) * 0.5
    labels = tf.ones((2, 4, 4, 2), tf.float32)
    dataset = tf.data.Dataset.from_tensors((images, labels))
    augmented = MultipleColorJitter(
        dataset,
        brightness=0.2,
        n_max=1,
        prob=1.0,
    )

    augmented_images, augmented_labels = next(iter(augmented()))

    assert augmented_images.shape == images.shape
    assert tf.reduce_all(augmented_images[1] == images[1]).numpy()
    assert tf.reduce_all(augmented_labels == labels).numpy()
