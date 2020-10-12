import tensorflow as tf


class MultipleColorJitter():
    def __init__(self, ds, brightness=0, contrast=0, saturation=0, hue=0, n_max=None, prob=0.2):
        self.dataset = ds
        self.prob = prob
        self.n_max = n_max

    def jitter(self, x, y):
        if tf.random.uniform([], 0, 1).numpy() <= self.prob:
            x = tf.image.adjust_brightness(x,)
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
