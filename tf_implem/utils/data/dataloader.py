import os
import tensorflow as tf
import numpy as np

# todo: test ocrdataloader


class SegmentationDataLoader():
    def __init__(self, path, n_classes=2):
        """ Loads the tf.data.Dataset 
        Parameters:
            -path        : Path of the Folder containing , 
                            images and labels folder
            -n_classes   : number of classes in labels.

        """

        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels")

        img_paths, label_paths = self._read_img_paths_and_labels(
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

    def _read_img_paths_and_labels(self, images_path, labels_path):
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


class OCRDataLoader():
    def __init__(self, path, image_width):
        """ Loads the tf.data.Dataset 
        Parameters:
            -annotation_path :  Path of  folder containing images folder,labels.txt and table.txt to be loaded in dataset
            -image_width : image width 

        """
        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels.txt")
        table_path = os.path.join(path, "table.txt")
        img_paths, labels = self._read_img_paths_and_labels(
            images_path,
            labels_path)
        self.image_width = image_width
        self.size = len(img_paths)
        self.path = path

        with open(table_path) as f:
            self.inv_table = [char.strip() for char in f]
        self.n_classes = len(self.inv_table)
        self.blank_index = self.n_classes - 1

        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), self.blank_index)

        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        ds = ds.map(self._decode_and_resize)
        ds = ds.apply(tf.data.experimental.ignore_errors())

        self.dataset = ds

    def _decode_and_resize(self, filename, label):
        """ Reads and resize image (64,image_width)
            Parameters:
                -filename : Name of file.
                -label   : label of that image.

            returns image tensor and label.
        """
        image = tf.io.read_file(os.path.join(self.path, filename))
        image = tf.io.decode_png(image, channels=1)
        image = 1.0-tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (64, self.image_width))
        return image, label

    def _convert_label(self, image, label):
        """ maps chars in label to integers  according to table.txt

            Parameters:
                -image; Image tensor
                -label: label

            returns image tensor and label sparse tensor
        """
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.table.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)
        return image, sparse_label

    def __len__(self):
        return self.size

    def __call__(self, batch_size=8, repeat=1):
        """Return tf.data.Dataset."""
        ds = self.dataset.batch(batch_size).map(
            self._convert_label).repeat(repeat)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = ds
        return self.dataset

    def _read_img_paths_and_labels(self, data_path, labels_path):
        """return image filenames  and respective labels.
        Parameters:
            -data_path : Path of the folder of images.
            -labels_path : path of label.txt

        """
        img_path = os.listdir(data_path)
        label = open(labels_path, 'r')
        labels = label.readlines()
        labels = [labels[i].split(":")[1].strip() for i in range(len(labels))]
        labels_ = []
        for i in img_path:
            labels_.append(labels[int(i.split(".")[0])])
        return img_path, labels_

    def map_to_chars(self, inputs, table, blank_index=0, merge_repeated=False):
        """Maps Integers to characters 
        Parameters:
            -input : sparse tensor to be converted.
            -table : tf.lookup.StaticHashTable acoording to which mapping is done.
            -blank_index : index saved for space default 0
            -merge_repeated : bollean indicated weather repeated integers are to merged or not.

            return list of strings
            """

        inputs = tf.sparse.to_dense(inputs, default_value=blank_index).numpy()
        lines = []
        for line in inputs:
            text = ""
            previous_char = -1
            for char_index in line:
                if merge_repeated:
                    if char_index == previous_char:
                        continue
                previous_char = char_index
                if char_index == blank_index:
                    continue
                text += table[char_index]
            lines.append(text)
        return lines
