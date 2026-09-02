"""High-level TensorFlow document-segmentation interfaces."""

import cv2
import numpy as np
import tensorflow as tf

from .utils.data.dataloader import SegmentationDataLoader
from .utils.models.segmentation_model import SegmentationModel
from .utils.train import Train


class _SegmentationProcessor(Train):
    def __init__(self, n_classes=2, log_dir="logs"):
        super().__init__(model=SegmentationModel(n_classes), log_dir=log_dir)
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, ignore_errors=False):
        self.train_dl = SegmentationDataLoader(
            train_path,
            self.model.n_classes,
            ignore_errors=ignore_errors,
        )
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path,
                self.model.n_classes,
                ignore_errors=ignore_errors,
            )
        return self.train_dl, self.val_dl

    def train(
        self,
        n_epochs,
        train_dl=None,
        val_dl=None,
        batch_size=1,
        repeat=1,
        criterion=None,
        class_weights=None,
        optimizer=None,
        weight_decay=None,
        learning_rate=0.0001,
        lr_decay=None,
        save_checkpoints=True,
        checkpoint_freq=None,
        checkpoint_path=None,
        max_to_keep=5,
        log_freq=100,
    ):
        train_dl = self.train_dl if train_dl is None else train_dl
        val_dl = self.val_dl if val_dl is None else val_dl
        return super().train(
            n_epochs,
            train_dl=train_dl,
            val_dl=val_dl,
            batch_size=batch_size,
            repeat=repeat,
            criterion=criterion,
            class_weights=class_weights,
            optimizer=optimizer,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            save_checkpoints=save_checkpoints,
            checkpoint_freq=checkpoint_freq,
            checkpoint_path=checkpoint_path,
            max_to_keep=max_to_keep,
            log_freq=log_freq,
        )

    def get_coordinates(self, probability_map, class_index=1, iterations=2):
        """Return area-sorted bounding boxes for one predicted class."""
        if not 0 <= class_index < self.model.n_classes:
            raise ValueError("class_index is outside the model output classes.")
        if iterations < 0:
            raise ValueError("iterations cannot be negative.")

        probability_map = tf.convert_to_tensor(probability_map)
        if probability_map.shape.rank == 4:
            if probability_map.shape[0] not in (1, None):
                raise ValueError(
                    "Postprocessing currently accepts one image at a time."
                )
            probability_map = probability_map[0]
        if probability_map.shape.rank != 3:
            raise ValueError("probability_map must have shape [H, W, classes].")

        class_map = tf.argmax(probability_map, axis=-1, output_type=tf.int32)
        binary = tf.cast(tf.equal(class_map, class_index), tf.uint8) * 255
        binary = binary.numpy()
        if iterations:
            binary = cv2.erode(
                binary,
                np.ones((7, 7), dtype=np.uint8),
                iterations=iterations,
            )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        return boxes

    @staticmethod
    def _crop_boxes(original_image, boxes, map_shape):
        image_height, image_width = (int(value) for value in original_image.shape[:2])
        map_height, map_width = (
            int(value) for value in tf.convert_to_tensor(map_shape).numpy()[:2]
        )
        scale_x = image_width / map_width
        scale_y = image_height / map_height

        crops = []
        for x, y, width, height in boxes:
            left = max(0, round(x * scale_x))
            top = max(0, round(y * scale_y))
            right = min(image_width, round((x + width) * scale_x))
            bottom = min(image_height, round((y + height) * scale_y))
            crops.append(original_image[top:bottom, left:right])
        return crops


class PageExtractor(_SegmentationProcessor):
    """Segment and crop page regions from document images."""

    def __init__(self, log_dir="logs"):
        super().__init__(n_classes=2, log_dir=log_dir)

    def get_pages(
        self,
        original_image,
        probability_map,
        num_pages=None,
        min_area_percent=0.8,
        iterations=2,
    ):
        if not 0.0 <= min_area_percent <= 1.0:
            raise ValueError("min_area_percent must be between 0 and 1.")
        boxes = self.get_coordinates(
            probability_map,
            class_index=1,
            iterations=iterations,
        )
        if not boxes:
            return []
        if num_pages is None:
            maximum_area = boxes[0][2] * boxes[0][3]
            boxes = [
                box
                for box in boxes
                if box[2] * box[3] >= min_area_percent * maximum_area
            ]
        else:
            boxes = boxes[:num_pages]
        return self._crop_boxes(original_image, boxes, tf.shape(probability_map)[-3:-1])


class ImageExtractor(_SegmentationProcessor):
    """Segment and crop embedded images from document pages."""

    def __init__(self, log_dir="logs"):
        super().__init__(n_classes=2, log_dir=log_dir)

    def get_images(
        self,
        original_image,
        probability_map,
        num_images=None,
        iterations=2,
    ):
        boxes = self.get_coordinates(
            probability_map,
            class_index=1,
            iterations=iterations,
        )
        if num_images is not None:
            boxes = boxes[:num_images]
        return self._crop_boxes(original_image, boxes, tf.shape(probability_map)[-3:-1])


class LayoutExtractor(_SegmentationProcessor):
    """Segment multiple document-layout classes."""

    def __init__(self, n_layouts, log_dir="logs"):
        super().__init__(n_classes=n_layouts, log_dir=log_dir)

    def get_layouts(
        self,
        original_image,
        probability_map,
        class_index,
        num_regions=None,
        iterations=2,
    ):
        boxes = self.get_coordinates(
            probability_map,
            class_index=class_index,
            iterations=iterations,
        )
        if num_regions is not None:
            boxes = boxes[:num_regions]
        return self._crop_boxes(original_image, boxes, tf.shape(probability_map)[-3:-1])


class BaselineDetector(_SegmentationProcessor):
    """Segment text baselines from document images."""

    def __init__(self, log_dir="logs"):
        super().__init__(n_classes=2, log_dir=log_dir)
