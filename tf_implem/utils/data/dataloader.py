import os
import tensorflow as tf
import numpy as np

# todo: add ocrdataloader


class SegmentationDataLoader():
    def __init__(self, path, n_classes=2):
        """ Loads the tf.data.Dataset 
        Parameters:
            -path        : Path of the Folder containing , 
                            images and labels folder
            -n_classes   : number of classes in labels.
            -batch_size  : Batch size 
            -repeat      : The number of times we want data to repeat. e.g if repeat 1 data is iterted once.

        """

        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels")

        img_paths, label_paths = read_img_paths_and_labels(
            images_path,
            labels_path)
        self.n_classes = n_classes
        self.size = len(img_paths)

        ds = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
        ds = ds.map(self._decode_and_resize)
        ds = ds.apply(tf.data.experimental.ignore_errors())

        self.dataset = ds

    def _decode_and_resize(self, image_filename, label_filename):
        """ Reads and resize image (64,image_width)
            Parameters:
                -image_filename : Name of image file.
                -label_filename : Name of label file.

            returns image tensor and label tensor.
        """
        image = tf.io.read_file(image_filename)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        #image = tf.image.resize(image,self.resize)
        label = tf.io.read_file(label_filename)
        label = tf.io.decode_png(label, channels=3)
        #label = tf.image.resize(label,self.resize)
        label = tf.cast((label[:, :, 0] > 100), tf.int32)
        label = tf.one_hot(label, self.n_classes)
        return image, label

    def __call__(self, batch_size=64, repeat=1):
        """Return tf.data.Dataset."""
        ds = self.dataset.batch(batch_size).repeat(repeat)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = ds
        return self.dataset

    def __len__(self):
        return self.size


def read_img_paths_and_labels(images_path, labels_path):
    """return image filenames  and respective label filenames.
       Parameters:
           -images_path : Path of the folder of images.
           -labels_path : path of the folder of images.

    """
    img_path = os.listdir(images_path)
    img_path.sort()
    img_path = [str(os.path.join(images_path, i)) for i in img_path]
    label_path = os.listdir(labels_path)
    label_path.sort()
    label_path = [str(os.path.join(labels_path, i)) for i in label_path]

    return img_path, label_path
