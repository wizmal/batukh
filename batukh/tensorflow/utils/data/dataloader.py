import os
import tensorflow as tf
import numpy as np

# todo: test ocrdataloader


class SegmentationDataLoader():
    r""" Loads the :class:`tensorflow.data.Dataset` for :class:`~batukh.tensorflow.segmenter.PageExtractor`, 
    :class:`~batukh.tensorflow.segmenter.ImageExtractor`, :class:`~batukh.tensorflow.segmenter.LayoutExtractor` and 
    :class:`~batukh.tensorflow.segmenter.BaselineDetector` classes.

    Example

    .. code:: python

        >>> from batukh.tensorflow.utils.data.dataloader import SegmentationDataLoader
        >>> dl=SegmentationDataLoader("/data/",2)
        >>> for i,j in dl(batch_size=1):
        ...     print(i.shape,j.shape)
        ...     break

    Args:
        path (str)       : Path of the folder containing originals folder and labels folder to be loaded in dataset.
            Folder names must be as mentioned.
        n_classes (int)  : number of classes in label images.



    """

    def __init__(self, path, n_classes):

        images_path = os.path.join(path, "originals")
        labels_path = os.path.join(path, "labels")
        assert os.path.isdir(
            images_path), "Path does not contian originals folder."
        assert os.path.isdir(
            labels_path), "Path does not contian labels folder."

        img_paths, label_paths = self._read_img_label_paths(
            images_path,
            labels_path)
        self.n_classes = n_classes
        self.size = len(img_paths)

        ds = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
        ds = ds.map(self._decode_and_resize)
        ds = ds.apply(tf.data.experimental.ignore_errors())

        self.dataset = ds
        print("Dataset build....")

    def _decode_and_resize(self, image_filename, label_filename):
        r""" Reads images. Reads and one hot encodes labels.

        Args:
            image_filename (str) : Path of image file.
            label_filename (str) : Path of label file.

        Returns:
            :class:`tensorflow.Tensor`  : Image tensor.
            :class:`tensorflow.Tensor`  : Label tensor.
        """
        image = tf.io.read_file(image_filename)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        resize = (tf.shape(image[:, :, 0])//32)*32
        image = tf.image.resize(image, resize)
        label = tf.io.read_file(label_filename)
        label = tf.io.decode_png(label, channels=3)
        label = tf.image.resize(label, resize)
        label = tf.cast((label[:, :, 0] > 100), tf.int32)
        label = tf.one_hot(label, self.n_classes)
        return image, label

    def __call__(self, batch_size=1, repeat=1):
        r"""

        Args:
            batch_size (int,optional) : Batchsize of :class:`~tf.data.datset`. 
                Default value 1.
            repeat (int, optional)    : Specifies the number of times the dataset will  be iterated in one epoch.
                Default value 1


        Return:
            :class:`tensorflow.data.dataset`  : Dataloader.
        """
        ds = self.dataset
        ds = ds.batch(batch_size).repeat(repeat)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def __len__(self):
        r"""
        Return:
            int:lenght of dataset
        """
        return self.size

    def _read_img_label_paths(self, images_path, labels_path):
        r"""Reads paths of images and labels.


        Args:
            images_path (str) : Path of the folder of images.
            labels_path (str) : Path of the folder of labels.

        Returns:
            list  : List of image paths.
            list : List of label paths.

        """
        img_path = os.listdir(images_path)
        img_path.sort()
        label_path = os.listdir(labels_path)
        label_path.sort()
        check = [True if i.split('.')[0] == j.split(
            ".")[0] else False for i, j in zip(img_path, label_path)]
        assert tf.reduce_all(
            check), "Originals and Labels does not contain  images with same filenames."
        img_path = [str(os.path.join(images_path, i)) for i in img_path]
        label_path = [str(os.path.join(labels_path, i)) for i in label_path]

        return img_path, label_path


class OCRDataLoader():
    r""" Loads the :class:`~tensorflow.data.Dataset` for :class:`~batukh.tensorflow.ocr.OCR` class.

    Example
    .. code:: python

        >>> from batukh.tensorflow.utils.data.dataloader import OCRDataLoader
        >>> dl=OCRDataLoader("/data/",64)
        >>> for i,j in dl(1,1):
        ...     print(i.shape,j.values)
        ...     break

    Args:
        path (str)        :  Path of  folder containing images folder,labels.txt and table.txt to be loaded in dataset.Name of folders and files should be same as mentioned.
        height (int,optional)      : Specifies the height to which all images will be resized keeping same aspect ratio.
            Default: :math:`64`
    """

    def __init__(self, path, height=64):

        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels.txt")
        table_path = os.path.join(path, "table.txt")
        assert os.path.isdir(
            images_path), "Path does not contian images folder."
        assert os.path.exists(
            labels_path), "Path does not contian label.txt."
        assert os.path.exists(
            labels_path), "Path does not contian table.txt."
        img_paths, labels = self._read_img_paths_and_labels(
            images_path,
            labels_path)
        self.size = len(img_paths)
        self.path = path
        self.height = height

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
        print("Dataset build....")

    def _decode_and_resize(self, filename, label):
        r""" Reads image.

        Args:
            filename (str) : Name of file.
            label    (str) : Label of  image.

        Returns:
            :class:`tensorflow.Tensor` : Image Tensor
            :class:`tensorflow.tf.Tensor` :  Label tensor

        """
        image = tf.io.read_file(filename)
        image = tf.io.decode_png(image, channels=1)
        image = 1.0-tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, (self.height, tf.shape(image)[-2]), preserve_aspect_ratio=True)
        return image, label

    def _convert_label(self, image, label):
        r""" Maps chars in label to integers  according to table.txt

        Args:
            image ( :class:`tensorflow.tensor`) : Image tensor
            label  (str)      : label

        Returns:
            :class:`tensorflow.Tensor` : Image tensor. 
            :class:`tensorflow.Tensor` : Label sparse tensor.
        """
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.table.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        label = tf.cast(sparse_label, tf.int32)
        return image, label

    def __len__(self):
        r"""
        Return:
            int:lenght of dataset
        """
        return self.size

    def __call__(self, repeat=1):
        r"""

        Args:
            batch_size (int,optional): specifies the batch size.
                Default: :math:`1`
            repeat (int,optional): Specifes the of time :class:`tensorflow.data.dataset` will be itterated in one epoch.
                Default: :math:`1`
        Returns:
            :class:`tensorflow.data.dataset` : Dataloader.
            """
        ds = self.dataset
        ds = ds.batch(1).map(
            self._convert_label).repeat(repeat)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _read_img_paths_and_labels(self, images_path, labels_path):
        r"""reads filenames and respective labels.

        Args:
            imgages_path (str)  : Path of image folder.
            labels_path (str)  : path of label.txt
        Returns:
            list   : List of  images filenames.
            list      : List of labels.
        """
        img_path_ = os.listdir(images_path)
        img_path_.sort()

        label_ = open(labels_path, 'r')
        labels_ = label_.readlines()
        labels_ = [labels_[i].split(":")[1].strip()
                   for i in range(len(labels_))]
        labels = []
        check = [True if 0 <= int(i.split(".")[0]) < len(
            labels) else False for i in img_path_]
        img_path = [os.path.join(images_path, i) for i in img_path_]
        assert tf.reduce_all(
            check), "Image filenames doesnt match with any line number in labels.txt"

        for i in img_path:
            labels.append(labels_[int(i.split(".")[0].split("/")[-1])])
        return img_path, labels
