from .utils.models.segmentation_model import SegmentationModel
from .utils.train import Train
from .utils.data.dataloader import SegmentationDataLoader
from .utils.data.augmentation import MultipleColorJitter


# todo:apply augmentation


class PageExtractor(Train):
    r"""This class used to extract pages (removing borders and blank spaces around pages) from originals.

    Example

    .. code-block:: python

        >>> from batukh.tensorflow.segmenter import PageExtraction
        >>> page_Extractor = PageExtraction()
        >>> page_Extractor.load_data(train_path = "/train_data/")
        >>> page_Extractor.train(n_epochs=10,batch_size=1,weights=[1,100])
        Initializing from scratch
        Epoch: 1. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Model saved to /tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14280
        >>> page_Extractor.save_model("/model/")
        Model saved at /saved_models


        """

    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  pixels corresponding to page area colored red.

        Args:
            train_path (str)        : path of the folder contaings originals folder (containing original images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings originals folder (containing original images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, train_dl=None, val_dl=None, batch_size=1, repeat=1, criterion=None, class_weights=None, optimizer=None, weight_decay=None, learning_rate=0.0001, lr_decay=None, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5, log_freq=100):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl
        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, weight_decay=weight_decay, learning_rate=learning_rate, lr_decay=lr_decay, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep, log_freq=log_freq)

    def get_coordinates(self, probability_map, iterations=2):
        binary = tf.cast(
            (tf.math.argmax(probability_map, -1)*255), dtype=tf.uint8)
        kernel = tf.ones((7, 7))
        erroded = cv2.erode(binary.numpy(), kernel.numpy(),
                            iterations=iterations)
        (contours, _) = cv2.findContours(
            erroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.boundingRect(contour) for contour in contours]
        contours.sort(key=lambda x: x[-1]*x[-2], reverse=True)
        return contours

    def get_pages(self, original_image, probability_map, num_pages=None, min_area_percent=0.8, iterations=2):
        contours = self.get_coordinates(probability_map)
        if num_pages is None:
            max_area = contours[0][-1]*contours[0][-2]
            contours = [contour for contour in contours if contour[-1]
                        * contour[-2] >= min_area_percent*max_area]
        else:
            contours = contours[:num_pages]

        cropped_images = []
        for contour in contours:
            x, y, w, h = contour
            cropped_images.append(original_image[y:y+h, x:x+w])
        return cropped_images

class ImageExtractor(Train):
    r"""The class used to extract images.

    Example

    .. code-block:: python

        >>> from batukh.tensorflow.segmenter import ImageExtractor
        >>> image_Extractor = ImageExtractor()
        >>> image_Extractor.load_data( train_path="/train_data/",val_path="/val_data/")
        >>> image_Extractor.train(n_epochs=1)
        Initializing from scratch
        Epoch: 1. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Epoch: 1. validation: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Model saved to /tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14280

        """

    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  pixels corresponding image areas colored red.

        Args:
            train_path (str)        : path of the folder contaings originals folder (containing originals images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings originals folder (containing originals images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, train_dl=None, val_dl=None, batch_size=1, repeat=1, criterion=None, class_weights=None, optimizer=None, weight_decay=None, learning_rate=0.0001, lr_decay=None, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5, log_freq=100):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl

        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, weight_decay=weight_decay, learning_rate=learning_rate, lr_decay=lr_decay, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep, log_freq=log_freq)

    def get_coordinates(self, probability_map, iterations=2):
        binary = tf.cast(
            (tf.math.argmax(probability_map, -1)*255), dtype=tf.uint8)
        kernel = tf.ones((7, 7))
        erroded = cv2.erode(binary.numpy(), kernel.numpy(),
                            iterations=iterations)
        (contours, _) = cv2.findContours(
            erroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.boundingRect(contour) for contour in contours]
        contours.sort(key=lambda x: x[-1]*x[-2], reverse=True)
        return contours

    def get_images(self, original_image, probability_map, num_pages=None, iterations=2):
        contours = self.get_coordinates(probability_map,iterations)
        if num_pages is not None:
            contours = contours[:num_pages]

        cropped_images = []
        for contour in contours:
            x, y, w, h = contour
            cropped_images.append(original_image[y:y+h, x:x+w])
        return cropped_images


class LayoutExtractor(Train):
    r"""This class is used to extract diffrent layouts from a image.

    Example

    .. code-block:: python

        >>> from batukh.tensorflow.segmenter import LayoutExtractor
        >>> layout_Extractor = LayoutExtractor(2)
        >>> layout_Extractor.load_data(train_path ="/train_data/",val_data="/val_data/")
        >>> layout_Extractor.train(n_epochs=1,checkpoint_path="/tf_chkpts/")
        Restored from /tf_chkpts/ckpt-13280
        Epoch: 1. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Epoch: 1. validation: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Model saved to /tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14280

        """

    def __init__(self, n_layouts):
        super().__init__(model=SegmentationModel(n_layouts))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should bepython of same size with same filename.
            label images should be of black background with  pixels corresponding to diffrent areas colored diffrently.

        Args:
            train_path (str)        : path of the folder contaings originals folder (containing originals images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings originals folder (containing originals images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, train_dl=None, val_dl=None, batch_size=1, repeat=1, criterion=None, class_weights=None, optimizer=None, weight_decay=None, learning_rate=0.0001, lr_decay=None, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5, log_freq=None):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl
        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, weight_decay=weight_decay, learning_rate=learning_rate, lr_decay=lr_decay, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep, log_freq=log_freq)
        def get_coordinates(self, probability_map, iterations=2):
        binary = tf.cast(
            (tf.math.argmax(probability_map, -1)*255), dtype=tf.uint8)
        kernel = tf.ones((7, 7))
        erroded = cv2.erode(binary.numpy(), kernel.numpy(),
                            iterations=iterations)
        (contours, _) = cv2.findContours(
            erroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.boundingRect(contour) for contour in contours]
        contours.sort(key=lambda x: x[-1]*x[-2], reverse=True)
        return contours

    def get_layouts(self, original_image, probability_map, num_pages=None, iterations=2):
        contours = self.get_coordinates(probability_map,iterations)
        if num_pages is not None:
            contours = contours[:num_pages]

        cropped_images = []
        for contour in contours:
            x, y, w, h = contour
            cropped_images.append(original_image[y:y+h, x:x+w])
        return cropped_images


class BaselineDetector(Train):
    r"""This class is used to detect baseline.

    Example

    .. code-block:: python

        >>> from batukh.tensorflow.segmenter import BaselineDetector
        >>> baseline_Detector = BaselineDetector()
        >>> baseline_Detector.load_data("/train_data/")
        >>> baseline_Detector.train(1,weights=[1:700])
        Initializing from scratch
        Epoch: 1. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
        Model saved to /tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14280

        """

    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  red lines of about 5px representing baselines.

        Args:
            train_path (str)        : path of the folder contaings originals folder (containing original images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings originals folder (containing original images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classest)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, train_dl=None, val_dl=None, batch_size=1, repeat=1, criterion=None, class_weights=None, optimizer=None, weight_decay=None, learning_rate=0.0001, lr_decay=None, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5, log_freq=100):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl
        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, weight_decay=weight_decay, learning_rate=learning_rate, lr_decay=lr_decay, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep, log_freq=log_freq)
