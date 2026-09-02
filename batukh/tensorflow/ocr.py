"""High-level TensorFlow OCR interface."""

import warnings

import tensorflow as tf

from .utils.data.dataloader import OCRDataLoader
from .utils.models.ocr_model import OCRModel
from .utils.train import Train


class OCR(Train):
    """Train and run the custom CNN-BLSTM-CTC OCR model."""

    def __init__(self, n_classes, blank_index=None, log_dir="logs"):
        model = OCRModel(n_classes, blank_index=blank_index)
        super().__init__(
            model=model,
            is_ocr=True,
            blank_index=model.blank_index,
            log_dir=log_dir,
        )
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, height=64, ignore_errors=False):
        self.train_dl = OCRDataLoader(
            train_path,
            height=height,
            ignore_errors=ignore_errors,
        )
        self._validate_loader(self.train_dl)

        if val_path is not None:
            self.val_dl = OCRDataLoader(
                val_path,
                height=height,
                ignore_errors=ignore_errors,
            )
            self._validate_loader(self.val_dl)
            if self.val_dl.inv_table != self.train_dl.inv_table:
                raise ValueError("Training and validation character tables must match.")
        return self.train_dl, self.val_dl

    def _validate_loader(self, loader):
        if loader.n_classes != self.model.n_classes:
            raise ValueError(
                "OCR n_classes does not match table.txt: "
                f"model={self.model.n_classes}, table={loader.n_classes}."
            )
        if loader.blank_index != self.blank_index:
            raise ValueError(
                "The model and data loader must use the same CTC blank index."
            )

    def train(
        self,
        n_epochs,
        train_dl=None,
        val_dl=None,
        batch_size=1,
        repeat=1,
        criterion=None,
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

    def map2string(self, inputs, table=None, blank_index=None):
        """Map a sparse or dense batch of character IDs to Python strings."""
        if table is None:
            if self.train_dl is None:
                raise ValueError(
                    "table is required before load_data() has been called."
                )
            table = self.train_dl.inv_table
        if blank_index is None:
            blank_index = self.blank_index

        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs, default_value=blank_index)
        values = tf.convert_to_tensor(inputs).numpy()
        if values.ndim == 1:
            values = values[None, :]

        strings = []
        for sequence in values:
            characters = [
                table[int(index)] for index in sequence if int(index) != blank_index
            ]
            strings.append("".join(characters))
        return strings

    def decode(
        self,
        inputs,
        from_pred=True,
        method="greedy",
        merge_repeated=True,
        table=None,
        blank_index=None,
        beam_width=100,
        sequence_length=None,
    ):
        """Decode model logits or pre-decoded indices into strings."""
        if blank_index is None:
            blank_index = self.blank_index
        if method == "gready":
            warnings.warn(
                "'gready' is deprecated; use 'greedy'.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = "greedy"

        decoded = inputs
        if from_pred:
            logits = tf.convert_to_tensor(inputs)
            if logits.shape.rank != 3:
                raise ValueError("CTC logits must have shape [batch, time, classes].")
            if sequence_length is None:
                sequence_length = tf.fill(
                    (tf.shape(logits)[0],),
                    tf.shape(logits)[1],
                )
            else:
                sequence_length = tf.minimum(
                    tf.cast(sequence_length, tf.int32),
                    tf.shape(logits)[1],
                )
            time_major_logits = tf.transpose(logits, perm=(1, 0, 2))

            if method == "greedy":
                paths, _ = tf.nn.ctc_greedy_decoder(
                    inputs=time_major_logits,
                    sequence_length=sequence_length,
                    merge_repeated=merge_repeated,
                    blank_index=blank_index,
                )
            elif method == "beam_search":
                if blank_index != self.model.n_classes - 1:
                    raise ValueError(
                        "TensorFlow beam search requires the final class to be CTC blank."
                    )
                paths, _ = tf.nn.ctc_beam_search_decoder(
                    inputs=time_major_logits,
                    sequence_length=sequence_length,
                    beam_width=beam_width,
                    top_paths=1,
                )
            else:
                raise ValueError("method must be 'greedy' or 'beam_search'.")
            decoded = paths[0]

        return self.map2string(
            decoded,
            table=table,
            blank_index=blank_index,
        )
