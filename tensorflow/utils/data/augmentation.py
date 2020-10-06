import tensorflow as tf
import numpy as np


class MultipleColorJitter():
    def __init__(self, ds, prob=0.02):
        self.dataset = ds
        self.prob = prob

    def jitter(self, x, y):
        if np.random.choice(np.linspace(start=0.0, stop=1.0, num=100)) <= self.prob:
            x = tf.image.adjust_brightness(x, np.random.choice(
                np.linspace(start=0.0, stop=1.0, num=100)))
            x = tf.image.adjust_contrast(x, np.random.choice(
                np.linspace(start=0.0, stop=1.0, num=100)))
            x = tf.image.adjust_hue(x, np.random.choice(
                np.linspace(start=0.0, stop=1.0, num=100)))
            x = tf.image.adjust_saturation(x, np.random.choice(
                np.linspace(start=0.0, stop=1.0, num=100)))
        return (x, y)

    def __call__(self):
        return self.dataset().map(self.jitter)

    # todo:list nmax,arguments add
