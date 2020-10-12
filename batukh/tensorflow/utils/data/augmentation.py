import tensorflow as tf


class MultipleColorJitter():
    def __init__(self, ds, brightness=0, contrast=0, saturation=0, hue=0, n_max=None, prob=0.2):
        self.dataset = ds
        self.prob = prob
        self.n_max = n_max
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def jitter(self, batch):
        if tf.random.uniform([], 0, 1) <= self.prob:
            for i in range(self.n_max):

                batch[i] = tf.image.adjust_brightness(
                    batch[i], tf.random.uniform([], -1*self.brightness, self.brightness))
                batch[i] = tf.image.adjust_contrast(
                    batch[i], tf.random.uniform([], -1*self.contrast, self.contrast))
                batch[i] = tf.image.adjust_hue(
                    batch[i], tf.random.uniform([], -1*self.hue, self.hue))
                batch[i] = tf.image.adjust_saturation(
                    batch[i], tf.random.uniform([], -1*self.saturation, self.saturation))
        return batch

    def __call__(self):
        return self.dataset().map(self.jitter)
    # todo:list nmax,arguments add
