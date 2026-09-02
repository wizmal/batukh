"""TensorFlow ``tf.data`` input pipelines for segmentation and OCR."""

from pathlib import Path

import tensorflow as tf


def _files_in(directory):
    return sorted(path for path in Path(directory).iterdir() if path.is_file())


class SegmentationDataLoader:
    """Load paired document images and segmentation masks.

    ``path`` must contain ``originals`` and ``labels`` directories. Files are
    paired by filename stem. Binary masks use red-channel values above 100;
    multiclass masks store zero-based class IDs in the red channel.
    """

    def __init__(self, path, n_classes, ignore_errors=False):
        self.path = Path(path)
        self.n_classes = int(n_classes)
        self.ignore_errors = bool(ignore_errors)
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2.")

        images_path = self.path / "originals"
        labels_path = self.path / "labels"
        if not images_path.is_dir():
            raise FileNotFoundError(f"Missing originals directory: {images_path}")
        if not labels_path.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {labels_path}")

        image_paths, label_paths = self._read_img_label_paths(images_path, labels_path)
        self.size = len(image_paths)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.map(
            self._decode_and_resize,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if self.ignore_errors:
            dataset = dataset.ignore_errors(log_warning=True)
        self.dataset = dataset

    def _decode_and_resize(self, image_filename, label_filename):
        image = tf.io.decode_image(
            tf.io.read_file(image_filename),
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.convert_image_dtype(image, tf.float32)

        image_size = tf.shape(image)[:2]
        target_size = tf.maximum((image_size // 32) * 32, 32)
        image = tf.image.resize(
            image,
            target_size,
            method="bilinear",
            antialias=True,
        )

        label = tf.io.decode_image(
            tf.io.read_file(label_filename),
            channels=3,
            expand_animations=False,
        )
        label.set_shape((None, None, 3))
        label = tf.image.resize(
            tf.cast(label, tf.float32),
            target_size,
            method="nearest",
        )

        if self.n_classes == 2:
            label_ids = tf.cast(label[..., 0] > 100.0, tf.int32)
        else:
            label_ids = tf.cast(tf.round(label[..., 0]), tf.int32)
            tf.debugging.assert_greater_equal(
                label_ids,
                0,
                message="Segmentation masks must use non-negative class IDs.",
            )
            tf.debugging.assert_less(
                label_ids,
                self.n_classes,
                message="A mask class ID is outside n_classes.",
            )

        return image, tf.one_hot(label_ids, self.n_classes, dtype=tf.float32)

    def __call__(
        self,
        batch_size=1,
        repeat=1,
        shuffle=False,
        drop_remainder=False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        dataset = self.dataset
        if shuffle and self.size > 1:
            dataset = dataset.shuffle(self.size, reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape((None, None, 3)),
                tf.TensorShape((None, None, self.n_classes)),
            ),
            padding_values=(
                tf.constant(0.0, tf.float32),
                tf.constant(0.0, tf.float32),
            ),
            drop_remainder=drop_remainder,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def __len__(self):
        return self.size

    @staticmethod
    def _read_img_label_paths(images_path, labels_path):
        image_files = _files_in(images_path)
        label_files = _files_in(labels_path)
        if not image_files:
            raise ValueError(f"No input images found in {images_path}")

        labels_by_stem = {}
        for label_path in label_files:
            if label_path.stem in labels_by_stem:
                raise ValueError(
                    f"Duplicate label stem '{label_path.stem}' in {labels_path}"
                )
            labels_by_stem[label_path.stem] = label_path

        missing = [path.name for path in image_files if path.stem not in labels_by_stem]
        image_stems = {path.stem for path in image_files}
        extras = [path.name for path in label_files if path.stem not in image_stems]
        if missing or extras:
            raise ValueError(
                "Original/label filenames do not match. "
                f"Missing labels: {missing}; extra labels: {extras}"
            )

        return (
            [str(path) for path in image_files],
            [str(labels_by_stem[path.stem]) for path in image_files],
        )


class OCRDataLoader:
    """Load variable-width OCR images and sparse Unicode labels.

    ``path`` must contain ``images``, ``labels.txt``, and ``table.txt``.
    The final line of ``table.txt`` is reserved for the CTC blank class.
    """

    def __init__(self, path, height=64, ignore_errors=False):
        self.path = Path(path)
        self.height = int(height)
        self.ignore_errors = bool(ignore_errors)
        if self.height < 2:
            raise ValueError("height must be at least 2.")

        images_path = self.path / "images"
        labels_path = self.path / "labels.txt"
        table_path = self.path / "table.txt"
        if not images_path.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing character table: {table_path}")

        image_paths, labels = self._read_img_paths_and_labels(images_path, labels_path)
        self.size = len(image_paths)

        with table_path.open(encoding="utf-8") as table_file:
            self.inv_table = table_file.read().splitlines()
        if len(self.inv_table) < 2:
            raise ValueError(
                "table.txt must contain at least one symbol and a final CTC blank."
            )
        if len(set(self.inv_table)) != len(self.inv_table):
            raise ValueError("table.txt contains duplicate entries.")

        self.n_classes = len(self.inv_table)
        self.blank_index = self.n_classes - 1
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                str(table_path),
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            default_value=tf.cast(self.blank_index, tf.int64),
        )

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(
            self._decode_resize_and_encode,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if self.ignore_errors:
            dataset = dataset.ignore_errors(log_warning=True)
        self.dataset = dataset

    def _decode_resize_and_encode(self, filename, label):
        image = tf.io.decode_image(
            tf.io.read_file(filename),
            channels=1,
            expand_animations=False,
        )
        image.set_shape((None, None, 1))
        image = 1.0 - tf.image.convert_image_dtype(image, tf.float32)

        source_size = tf.shape(image)[:2]
        target_width = tf.maximum(
            1,
            tf.cast(
                tf.math.round(
                    tf.cast(source_size[1], tf.float32)
                    * tf.cast(self.height, tf.float32)
                    / tf.cast(source_size[0], tf.float32)
                ),
                tf.int32,
            ),
        )
        image = tf.image.resize(
            image,
            (self.height, target_width),
            method="bilinear",
            antialias=True,
        )

        characters = tf.strings.unicode_split(label, input_encoding="UTF-8")
        encoded_label = tf.cast(self.table.lookup(characters), tf.int32)
        assertion = tf.debugging.assert_none_equal(
            encoded_label,
            tf.cast(self.blank_index, tf.int32),
            message="An OCR label contains a character missing from table.txt.",
        )
        with tf.control_dependencies((assertion,)):
            encoded_label = tf.identity(encoded_label)
        return image, encoded_label, target_width

    def __call__(
        self,
        batch_size=1,
        repeat=1,
        shuffle=False,
        drop_remainder=False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        dataset = self.dataset
        if shuffle and self.size > 1:
            dataset = dataset.shuffle(self.size, reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape((self.height, None, 1)),
                tf.TensorShape((None,)),
                tf.TensorShape(()),
            ),
            padding_values=(
                tf.constant(0.0, tf.float32),
                tf.cast(self.blank_index, tf.int32),
                tf.constant(0, tf.int32),
            ),
            drop_remainder=drop_remainder,
        )
        dataset = dataset.map(
            self._dense_labels_to_sparse,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _dense_labels_to_sparse(self, images, labels, image_widths):
        valid = tf.not_equal(labels, tf.cast(self.blank_index, labels.dtype))
        indices = tf.cast(tf.where(valid), tf.int64)
        values = tf.gather_nd(labels, indices)
        dense_shape = tf.cast(tf.shape(labels), tf.int64)
        sparse_labels = tf.SparseTensor(indices, values, dense_shape)
        # OCRModel has one same-padded stride-2 pooling operation on its time axis.
        logit_lengths = tf.math.floordiv(image_widths + 1, 2)
        return images, (sparse_labels, logit_lengths)

    def __len__(self):
        return self.size

    @staticmethod
    def _read_img_paths_and_labels(images_path, labels_path):
        label_by_id = {}
        with Path(labels_path).open(encoding="utf-8") as labels_file:
            for line_number, line in enumerate(labels_file, start=1):
                raw_line = line.rstrip("\r\n")
                if not raw_line:
                    continue
                if ":" not in raw_line:
                    raise ValueError(
                        f"Invalid labels.txt line {line_number}; expected '<id>: <text>'."
                    )
                image_id, text = raw_line.split(":", 1)
                image_id = image_id.strip()
                if image_id in label_by_id:
                    raise ValueError(f"Duplicate OCR label ID '{image_id}'.")
                label_by_id[image_id] = text.strip()

        def sort_key(path):
            return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

        image_files = sorted(_files_in(images_path), key=sort_key)
        if not image_files:
            raise ValueError(f"No OCR images found in {images_path}")

        missing = [path.name for path in image_files if path.stem not in label_by_id]
        if missing:
            raise ValueError(f"No labels.txt entry found for images: {missing}")

        return (
            [str(path) for path in image_files],
            [label_by_id[path.stem] for path in image_files],
        )
