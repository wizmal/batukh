"""TensorFlow-native augmentation helpers."""

import tensorflow as tf


class MultipleColorJitter:
    """Apply one random color jitter to the first images in each batch.

    The wrapper accepts either a ``tf.data.Dataset`` or a callable data loader.
    Labels, when present, pass through unchanged.
    """

    def __init__(
        self,
        dataset,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        n_max=None,
        probability=0.2,
        prob=None,
    ):
        if prob is not None:
            probability = prob
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        if brightness < 0.0 or contrast < 0.0 or saturation < 0.0:
            raise ValueError(
                "brightness, contrast, and saturation must be non-negative."
            )
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must be between 0 and 0.5.")
        if n_max is not None and n_max < 0:
            raise ValueError("n_max cannot be negative.")

        self.dataset = dataset
        self.probability = float(probability)
        self.n_max = n_max
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def jitter(self, images, labels=None):
        images = tf.convert_to_tensor(images)
        tf.debugging.assert_rank(
            images, 4, message="Color jitter expects batched images."
        )

        batch_size = tf.shape(images)[0]
        count = batch_size if self.n_max is None else tf.minimum(self.n_max, batch_size)

        def apply_jitter():
            selected = images[:count]
            selected = tf.image.adjust_brightness(
                selected,
                tf.random.uniform((), -self.brightness, self.brightness),
            )
            selected = tf.image.adjust_contrast(
                selected,
                tf.random.uniform(
                    (), max(0.0, 1.0 - self.contrast), 1.0 + self.contrast
                ),
            )
            selected = tf.image.adjust_hue(
                selected,
                tf.random.uniform((), -self.hue, self.hue),
            )
            selected = tf.image.adjust_saturation(
                selected,
                tf.random.uniform(
                    (), max(0.0, 1.0 - self.saturation), 1.0 + self.saturation
                ),
            )
            return tf.concat((tf.clip_by_value(selected, 0.0, 1.0), images[count:]), 0)

        augmented = tf.cond(
            tf.random.uniform(()) < self.probability,
            apply_jitter,
            lambda: images,
        )
        return augmented if labels is None else (augmented, labels)

    def __call__(self, *args, **kwargs):
        dataset = (
            self.dataset(*args, **kwargs) if callable(self.dataset) else self.dataset
        )
        return dataset.map(self.jitter, num_parallel_calls=tf.data.AUTOTUNE)
